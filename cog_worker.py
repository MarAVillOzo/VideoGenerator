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
from PIL import Image

from int8_loader import encode_pair

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RUNTIME = ROOT / "runtime"
LOCK = RUNTIME / "active_generation.pid"
COG_DIR = MODELS / "CogVideoX-5b-I2V"

WIDTH = 720
HEIGHT = 480
NUM_FRAMES = 49
FPS = 8


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
                raise RuntimeError(f"Ya existe una generación activa (PID {pid}).")
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


def center_crop(image: Image.Image, width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    image = image.convert("RGB")
    iw, ih = image.size
    target_ratio = width / height
    ratio = iw / ih
    if ratio > target_ratio:
        nw = int(ih * target_ratio)
        x0 = (iw - nw) // 2
        image = image.crop((x0, 0, x0 + nw, ih))
    else:
        nh = int(iw / target_ratio)
        y0 = (ih - nh) // 2
        image = image.crop((0, y0, iw, y0 + nh))
    return image.resize((width, height), Image.LANCZOS)


def write_mp4_atomic(frames, out_file: Path, fps: int = FPS):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name("__writing__" + out_file.name)
    tmp.unlink(missing_ok=True)
    try:
        with imageio.get_writer(
            str(tmp),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        ) as writer:
            for frame in frames:
                if isinstance(frame, Image.Image):
                    arr = np.asarray(frame.convert("RGB"))
                else:
                    arr = np.asarray(frame)
                    if arr.dtype != np.uint8:
                        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                writer.append_data(arr)

        size = tmp.stat().st_size
        if size <= 0:
            raise RuntimeError("FFmpeg creó un MP4 vacío.")
        os.replace(str(tmp), str(out_file))
        return out_file
    finally:
        tmp.unlink(missing_ok=True)


def model_files_ok():
    required = [
        COG_DIR / "transformer" / "config.json",
        COG_DIR / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        COG_DIR / "vae" / "config.json",
        COG_DIR / "vae" / "diffusion_pytorch_model.safetensors",
        COG_DIR / "scheduler" / "scheduler_config.json",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(
            "CogVideoX Premium incompleto. Faltan:\n" + "\n".join(missing)
        )


def build_pipeline():
    model_files_ok()

    # Keep the exact official BF16 CogVideoX weights.
    # low_cpu_mem_usage avoids creating an extra full-size copy while loading.
    say("[STAGE 3/6] Cargando CogVideoX-5B-I2V BF16 con bajo uso de RAM")

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        str(COG_DIR / "transformer"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        str(COG_DIR / "vae"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        str(COG_DIR / "scheduler")
    )

    pipe = CogVideoXImageToVideoPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )

    # Official low-VRAM path.
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_sequential_cpu_offload()

    return pipe


def cleanup_success(spec: dict, spec_path: Path):
    if bool(spec.get("keep_intermediates", False)):
        return
    try:
        spec_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        for p in Path(spec["output_path"]).glob("__writing__*.mp4"):
            p.unlink(missing_ok=True)
    except Exception:
        pass


def generate(spec: dict):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    if not spec.get("image_path"):
        raise RuntimeError("CogVideoX Premium requiere una imagen base.")

    image_path = Path(spec["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen base: {image_path}")

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    commit_available = vm.available + swap.free
    say(
        f"[INFO] RAM libre {vm.available/1024**3:.2f} GB | "
        f"pagefile libre {swap.free/1024**3:.2f} GB"
    )
    if commit_available < 13 * 1024**3:
        raise RuntimeError(
            "Hay muy poca RAM + memoria virtual disponible para CogVideoX. "
            "Cierra navegadores y aplicaciones antes de usar el motor Premium."
        )

    seed = int(spec["seed"])
    seed_everything(seed)
    steps = int(spec.get("cog_steps", 50))
    steps = max(20, min(50, steps))

    # Ampere-friendly speed settings; do not change model precision.
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    say("[STAGE 1/6] Cargando T5-XXL INT8 ya instalado")
    say("[STAGE 2/6] Creando embeddings CogVideoX (226 tokens) y liberando T5")

    prompt_embeds, prompt_mask, neg_embeds, neg_mask = encode_pair(
        spec["prompt"],
        spec.get("negative_prompt", ""),
        max_length=226,
    )

    # CogVideoX consumes the embeddings directly and doesn't need attention masks.
    prompt_embeds = prompt_embeds.detach().to(device="cpu", dtype=torch.bfloat16)
    neg_embeds = neg_embeds.detach().to(device="cpu", dtype=torch.bfloat16)
    del prompt_mask, neg_mask
    clear_cuda()

    pipe = build_pipeline()

    image = center_crop(Image.open(image_path), WIDTH, HEIGHT)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    counter = {"n": 0}

    def callback_on_step_end(_pipe, step, _timestep, callback_kwargs):
        counter["n"] = int(step) + 1
        say(f"[COG_STEP] {counter['n']}/{steps}")
        return callback_kwargs

    say(
        f"[STAGE 4/6] Generando CogVideoX Premium: "
        f"{WIDTH}x{HEIGHT}, {NUM_FRAMES} frames, {steps} pasos"
    )

    result = pipe(
        image=image,
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        num_videos_per_prompt=1,
        num_inference_steps=steps,
        num_frames=NUM_FRAMES,
        guidance_scale=6.0,
        use_dynamic_cfg=False,
        generator=generator,
        output_type="pil",
        max_sequence_length=226,
        callback_on_step_end=callback_on_step_end,
    )

    say("[STAGE 5/6] Codificando MP4")
    frames = result.frames[0]

    out_dir = Path(spec["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"cogpremium_{seed}_720x480_49f.mp4"
    write_mp4_atomic(frames, out_file, FPS)

    say(f"[STAGE 6/6] VIDEO LISTO: {out_file}")

    del result, frames, pipe, prompt_embeds, neg_embeds
    clear_cuda()
    return out_file


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Uso: cog_worker.py <job.json>")

    acquire_lock()
    spec_path = Path(sys.argv[1])
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        generate(spec)
        cleanup_success(spec, spec_path)
    except torch.OutOfMemoryError as e:
        clear_cuda()
        raise RuntimeError(
            "CogVideoX agotó la VRAM. Cierra cualquier programa que use la GPU "
            "y vuelve a probar. Este motor usa offload secuencial + VAE tiling."
        ) from e
    finally:
        release_lock()
        clear_cuda()


if __name__ == "__main__":
    main()
