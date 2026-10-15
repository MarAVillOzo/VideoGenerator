from __future__ import annotations

import gc
import json
import os
import random
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import psutil
import torch
from PIL import Image, ImageFilter
from safetensors import safe_open

from int8_loader import encode_pair

from ltx_video.inference import create_transformer
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RUNTIME = ROOT / "runtime"
LOCK = RUNTIME / "active_generation.pid"

CHECKPOINT = MODELS / "ltxv-2b-0.9.8-distilled.safetensors"

# CLEAN/STABLE v3.5 keeps the reliable single-pass schedule and adds 10s long clips
# through segmented generation and final stitching.
DISTILLED_TIMESTEPS = [1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]
SEGMENT_FRAMES = 49  # ~2 seconds @ 24 FPS
LONG_TARGET_FRAMES = 240  # exact 10 seconds @ 24 FPS


def say(text: str):
    print(text, flush=True)


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def acquire_lock():
    RUNTIME.mkdir(exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text(encoding="utf-8").strip())
            if psutil.pid_exists(pid):
                raise RuntimeError(
                    f"Ya existe una generación activa (PID {pid}). "
                    "Espera a que termine antes de iniciar otra."
                )
        except ValueError:
            pass
        LOCK.unlink(missing_ok=True)
    LOCK.write_text(str(os.getpid()), encoding="utf-8")


def release_lock():
    try:
        if LOCK.exists() and LOCK.read_text(encoding="utf-8").strip() == str(os.getpid()):
            LOCK.unlink()
    except Exception:
        pass


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_allowed_steps():
    try:
        with safe_open(str(CHECKPOINT), framework="pt") as f:
            metadata = f.metadata() or {}
        raw = metadata.get("config")
        if not raw:
            return None
        cfg = json.loads(raw)
        return cfg.get("allowed_inference_steps")
    except Exception:
        return None


def center_crop_and_resize(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    image = image.convert("RGB")
    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height

    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    image = image.resize((target_width, target_height), Image.LANCZOS)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    return image


def load_conditioning_image(image_path: str, width: int, height: int) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = center_crop_and_resize(img, width, height)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = (arr * 2.0) - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).contiguous()
    return tensor


def build_single_pass_pipeline():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"No existe el modelo LTX: {CHECKPOINT}")

    say("[STAGE 3/6] Cargando LTX-Video 2B en RTX 3060")
    vae = CausalVideoAutoencoder.from_pretrained(str(CHECKPOINT)).to(torch.bfloat16)
    transformer = create_transformer(str(CHECKPOINT), "bfloat16")
    scheduler = RectifiedFlowScheduler.from_pretrained(str(CHECKPOINT))
    patchifier = SymmetricPatchifier(patch_size=1)

    pipe = LTXVideoPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        patchifier=patchifier,
        prompt_enhancer_image_caption_model=None,
        prompt_enhancer_image_caption_processor=None,
        prompt_enhancer_llm_model=None,
        prompt_enhancer_llm_tokenizer=None,
        allowed_inference_steps=read_allowed_steps(),
    )

    pipe = pipe.to("cuda")
    say(
        "[INFO] LTX listo. VRAM PyTorch: "
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
    )
    return pipe


def tensor_to_frames(images: torch.Tensor, target_w: int, target_h: int, render_w: int, render_h: int, frames: int):
    images = images[:, :, :frames, :render_h, :render_w]

    if (render_h, render_w) != (target_h, target_w):
        b, c, f, h, w = images.shape
        x = images.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        x = torch.nn.functional.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        images = x.reshape(b, f, c, target_h, target_w).permute(0, 2, 1, 3, 4)

    arr = images[0].permute(1, 2, 3, 0).detach().cpu().float().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return arr


def write_mp4_from_frames(frames: np.ndarray, out_file: Path, fps: int):
    """
    Publicación atómica del MP4:
    primero escribe __writing__*.mp4, cierra FFmpeg, verifica tamaño y
    después renombra al archivo final. La interfaz nunca recibe un MP4 a medio escribir.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_name("__writing__" + out_file.name)
    tmp_file.unlink(missing_ok=True)

    try:
        with imageio.get_writer(
            str(tmp_file),
            fps=int(fps),
            codec="libx264",
            quality=8,
            macro_block_size=16,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)

        tmp_size = tmp_file.stat().st_size
        if tmp_size <= 0:
            raise RuntimeError(f"FFmpeg creó un MP4 vacío: {tmp_file}")

        os.replace(str(tmp_file), str(out_file))

        if not out_file.exists() or out_file.stat().st_size <= 0:
            raise RuntimeError(f"El MP4 final no quedó válido: {out_file}")

        return out_file
    finally:
        tmp_file.unlink(missing_ok=True)


def save_last_frame_png(frames: np.ndarray, out_file: Path):
    imageio.imwrite(str(out_file), frames[-1])
    return out_file


def build_conditioning_items_from_path(image_path: str | None, render_w: int, render_h: int, strength: float):
    if not image_path:
        return None
    p = Path(str(image_path))
    if not p.exists():
        raise FileNotFoundError(f"No existe la imagen guía: {image_path}")
    media_tensor = load_conditioning_image(str(p), render_w, render_h)
    strength = max(0.0, min(1.0, float(strength)))
    return [ConditioningItem(media_tensor, 0, strength)]


def run_single_pass_segment(
    spec: dict,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    neg_embeds: torch.Tensor,
    neg_mask: torch.Tensor,
    render_w: int,
    render_h: int,
    render_f: int,
    fps: int,
    segment_seed: int,
    conditioning_image_path: str | None,
    conditioning_strength: float,
    global_step_offset: int,
    global_total_steps: int,
):
    pipe = build_single_pass_pipeline()

    tr_param = next(pipe.transformer.parameters())
    device = tr_param.device
    dtype = tr_param.dtype

    pe = prompt_embeds.to(device=device, dtype=dtype)
    ne = neg_embeds.to(device=device, dtype=dtype)
    pm = prompt_mask.to(device=device, dtype=torch.bool)
    nm = neg_mask.to(device=device, dtype=torch.bool)

    conditioning_items = build_conditioning_items_from_path(
        conditioning_image_path, render_w, render_h, conditioning_strength
    )
    if conditioning_items:
        say(f"[INFO] Conditioning image: {conditioning_image_path}")
        say(f"[INFO] Fuerza de imagen: {conditioning_strength:.2f}")

    generator = torch.Generator(device="cpu").manual_seed(segment_seed)
    local_counter = {"n": 0}

    def callback(_pipe, _step, _timestep, _kwargs):
        local_counter["n"] += 1
        global_step = global_step_offset + local_counter["n"]
        say(f"[LTX_STEP] {global_step}/{global_total_steps}")

    human_protection = bool(spec.get("human_protection", False))
    extreme_fidelity = bool(spec.get("extreme_fidelity", False))
    if extreme_fidelity:
        decode_noise_scale = 0.015
        image_cond_noise_scale = 0.06
    elif human_protection:
        decode_noise_scale = 0.02
        image_cond_noise_scale = 0.10
    else:
        decode_noise_scale = 0.025
        image_cond_noise_scale = 0.15

    result = pipe(
        height=render_h,
        width=render_w,
        num_frames=render_f,
        frame_rate=fps,
        prompt=None,
        negative_prompt=None,
        timesteps=DISTILLED_TIMESTEPS,
        guidance_scale=1,
        stg_scale=0,
        rescaling_scale=1,
        skip_block_list=[42],
        skip_layer_strategy=SkipLayerStrategy.AttentionValues,
        generator=generator,
        prompt_embeds=pe,
        prompt_attention_mask=pm,
        negative_prompt_embeds=ne,
        negative_prompt_attention_mask=nm,
        output_type="pt",
        callback_on_step_end=callback,
        conditioning_items=conditioning_items,
        decode_timestep=0.05,
        decode_noise_scale=decode_noise_scale,
        mixed_precision=False,
        offload_to_cpu=True,
        enhance_prompt=False,
        stochastic_sampling=False,
        media_items=None,
        is_video=True,
        vae_per_channel_normalize=True,
        image_cond_noise_scale=image_cond_noise_scale,
    )

    frames = tensor_to_frames(
        result.images,
        target_w=int(spec["width"]),
        target_h=int(spec["height"]),
        render_w=render_w,
        render_h=render_h,
        frames=int(spec.get("segment_output_frames", SEGMENT_FRAMES)),
    )

    del result, pipe, pe, ne, pm, nm, conditioning_items
    clear_cuda()
    return frames


def save_single_pass_mp4(images: torch.Tensor, spec: dict, render_width: int, render_height: int):
    target_w = int(spec["width"])
    target_h = int(spec["height"])
    frames = int(spec["num_frames"])

    out_dir = Path(spec["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = spec.get("generation_mode", "text_to_video")
    prefix = "img2vid" if mode == "image_to_video" else "video"
    out_file = out_dir / (
        f"{prefix}_{int(spec['seed'])}_{target_w}x{target_h}_{frames}f.mp4"
    )

    arr = tensor_to_frames(images, target_w, target_h, render_width, render_height, frames)
    return write_mp4_from_frames(arr, out_file, int(spec["frame_rate"]))


def generate_standard(spec: dict):
    width = int(spec["width"])
    height = int(spec["height"])
    frames = int(spec["num_frames"])
    fps = int(spec["frame_rate"])
    seed = int(spec["seed"])

    render_w = ((width - 1) // 32 + 1) * 32
    render_h = ((height - 1) // 32 + 1) * 32
    render_f = ((frames - 2) // 8 + 1) * 8 + 1

    say(f"[INFO] Render estable SINGLE-PASS: {render_w}x{render_h}x{render_f}")

    say("[STAGE 1/6] Cargando T5-XXL INT8 local")
    say("[STAGE 2/6] Calculando embeddings y liberando T5")
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = encode_pair(
        spec["prompt"],
        spec.get("negative_prompt", ""),
        max_length=256,
    )
    say(
        "[INFO] T5 liberado. VRAM PyTorch: "
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
    )

    pipe = build_single_pass_pipeline()

    tr_param = next(pipe.transformer.parameters())
    device = tr_param.device
    dtype = tr_param.dtype

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    neg_embeds = neg_embeds.to(device=device, dtype=dtype)
    prompt_mask = prompt_mask.to(device=device, dtype=torch.bool)
    neg_mask = neg_mask.to(device=device, dtype=torch.bool)

    conditioning_items = build_conditioning_items_from_path(
        spec.get("image_path"), render_w, render_h, float(spec.get("image_strength", 0.95))
    )

    human_protection = bool(spec.get("human_protection", False))
    extreme_fidelity = bool(spec.get("extreme_fidelity", False))
    if extreme_fidelity:
        decode_noise_scale = 0.015
        image_cond_noise_scale = 0.06
    elif human_protection:
        decode_noise_scale = 0.02
        image_cond_noise_scale = 0.10
    else:
        decode_noise_scale = 0.025
        image_cond_noise_scale = 0.15

    generator = torch.Generator(device="cpu").manual_seed(seed)
    step_total = len(DISTILLED_TIMESTEPS)
    step_counter = {"n": 0}

    def callback(_pipe, _step, _timestep, _kwargs):
        step_counter["n"] += 1
        say(f"[LTX_STEP] {step_counter['n']}/{step_total}")

    say("[STAGE 4/6] Generando video SINGLE-PASS")
    result = pipe(
        height=render_h,
        width=render_w,
        num_frames=render_f,
        frame_rate=fps,
        prompt=None,
        negative_prompt=None,
        timesteps=DISTILLED_TIMESTEPS,
        guidance_scale=1,
        stg_scale=0,
        rescaling_scale=1,
        skip_block_list=[42],
        skip_layer_strategy=SkipLayerStrategy.AttentionValues,
        generator=generator,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_mask,
        negative_prompt_embeds=neg_embeds,
        negative_prompt_attention_mask=neg_mask,
        output_type="pt",
        callback_on_step_end=callback,
        conditioning_items=conditioning_items,
        decode_timestep=0.05,
        decode_noise_scale=decode_noise_scale,
        mixed_precision=False,
        offload_to_cpu=True,
        enhance_prompt=False,
        stochastic_sampling=False,
        media_items=None,
        is_video=True,
        vae_per_channel_normalize=True,
        image_cond_noise_scale=image_cond_noise_scale,
    )

    say("[STAGE 5/6] Codificando MP4")
    out = save_single_pass_mp4(result.images, spec, render_w, render_h)
    say(f"[STAGE 6/6] VIDEO LISTO: {out}")

    del result, pipe, prompt_embeds, neg_embeds, prompt_mask, neg_mask, conditioning_items
    clear_cuda()
    return out


def generate_long_10s(spec: dict):
    width = int(spec["width"])
    height = int(spec["height"])
    fps = int(spec["frame_rate"])
    target_frames = int(spec["num_frames"])

    render_w = ((width - 1) // 32 + 1) * 32
    render_h = ((height - 1) // 32 + 1) * 32
    render_f = SEGMENT_FRAMES

    segments = 5
    global_total_steps = segments * len(DISTILLED_TIMESTEPS)
    out_dir = Path(spec["output_path"])
    segments_dir = out_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    say(f"[INFO] Render 10s segmentado estable: {segments} segmentos de {SEGMENT_FRAMES} frames")
    say("[STAGE 1/6] Cargando T5-XXL INT8 local")
    say("[STAGE 2/6] Calculando embeddings y liberando T5")
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = encode_pair(
        spec["prompt"],
        spec.get("negative_prompt", ""),
        max_length=256,
    )
    say(
        "[INFO] T5 liberado. VRAM PyTorch: "
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
    )

    say("[STAGE 4/6] Generando clip largo segmentado")
    all_frames = []
    current_conditioning_path = spec.get("image_path")
    base_strength = float(spec.get("image_strength", 0.95))

    for seg_idx in range(segments):
        say(f"[SEGMENT] {seg_idx + 1}/{segments}")
        seg_seed = int(spec["seed"]) + seg_idx * 9973

        frames = run_single_pass_segment(
            spec=spec | {"segment_output_frames": SEGMENT_FRAMES},
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            neg_embeds=neg_embeds,
            neg_mask=neg_mask,
            render_w=render_w,
            render_h=render_h,
            render_f=render_f,
            fps=fps,
            segment_seed=seg_seed,
            conditioning_image_path=current_conditioning_path,
            conditioning_strength=base_strength if seg_idx == 0 else (0.98 if bool(spec.get('extreme_fidelity', False)) else (0.96 if bool(spec.get('human_protection', False)) else 0.92)),
            global_step_offset=seg_idx * len(DISTILLED_TIMESTEPS),
            global_total_steps=global_total_steps,
        )

        segment_path = segments_dir / f"segment_{seg_idx + 1:02d}.mp4"
        write_mp4_from_frames(frames, segment_path, fps)

        last_frame_path = segments_dir / f"segment_{seg_idx + 1:02d}_last.png"
        save_last_frame_png(frames, last_frame_path)
        current_conditioning_path = str(last_frame_path)

        all_frames.append(frames)

    say("[STAGE 5/6] Uniendo segmentos y codificando MP4 final")
    full = np.concatenate(all_frames, axis=0)
    full = full[:target_frames]  # exact 10 seconds @ 24 fps

    mode = spec.get("generation_mode", "text_to_video")
    prefix = "img2vid10s" if mode == "image_to_video" else "video10s"
    out_file = out_dir / (
        f"{prefix}_{int(spec['seed'])}_{width}x{height}_{target_frames}f.mp4"
    )
    out = write_mp4_from_frames(full, out_file, fps)
    say(f"[STAGE 6/6] VIDEO LISTO: {out}")

    del prompt_embeds, prompt_mask, neg_embeds, neg_mask, all_frames, full
    clear_cuda()
    return out



def cleanup_success_artifacts(spec: dict, spec_path: Path | None = None):
    """
    Solo se llama tras un render exitoso.
    Si keep_intermediates=False:
      - elimina outputs/<job>/segments
      - elimina el JSON de runtime del job
      - elimina __writing__*.mp4 abandonados dentro del job
    Conserva:
      - MP4 final
      - metadata.json (la escribe app.py)
      - copia input_image.* si existe
    """
    if bool(spec.get("keep_intermediates", False)):
        say("[CLEAN] Conservando archivos intermedios por solicitud del usuario")
        return

    out_dir = Path(spec["output_path"])
    segments_dir = out_dir / "segments"

    if segments_dir.exists():
        shutil.rmtree(segments_dir, ignore_errors=True)
        say("[CLEAN] Segmentos temporales eliminados")

    for p in out_dir.glob("__writing__*.mp4"):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    if spec_path is not None:
        try:
            spec_path.unlink(missing_ok=True)
            say("[CLEAN] JSON temporal de runtime eliminado")
        except Exception:
            pass


def generate_once(spec: dict):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    free_ram = psutil.virtual_memory().available / 1024**3
    if free_ram < 5.0:
        raise RuntimeError(
            f"Solo hay {free_ram:.2f} GB de RAM libre. "
            "Cierra Chrome/Epic/otras aplicaciones y vuelve a intentar."
        )

    seed = int(spec["seed"])
    seed_everything(seed)
    say(f"[INFO] RAM libre: {free_ram:.2f} GB")
    if bool(spec.get("human_protection", False)):
        say("[INFO] Protección humana / estética ACTIVADA")
    if bool(spec.get("extreme_fidelity", False)):
        say("[INFO] Fidelidad extrema / retrato premium ACTIVADA")

    if int(spec["num_frames"]) >= LONG_TARGET_FRAMES:
        return generate_long_10s(spec)
    return generate_standard(spec)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Uso: worker.py <job.json>")

    acquire_lock()
    try:
        spec_path = Path(sys.argv[1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        generate_once(spec)
        cleanup_success_artifacts(spec, spec_path)
    except torch.OutOfMemoryError as e:
        clear_cuda()
        raise RuntimeError(
            "CUDA se quedó sin VRAM. Cierra navegadores/juegos que usen GPU y "
            "prueba Draft 16:9 / 2 seconds."
        ) from e
    finally:
        release_lock()
        clear_cuda()


if __name__ == "__main__":
    main()
