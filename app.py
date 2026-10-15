from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import psutil
import requests

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RUNTIME = ROOT / "runtime"
MODELS = ROOT / "models"
OUTPUTS.mkdir(exist_ok=True)
RUNTIME.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

COG_MODEL_DIR = MODELS / "CogVideoX-5b-I2V"

ENGINES = [
    "⚡ LTX 2B FAST",
    "👑 CogVideoX 5B PREMIUM (I2V • 6s • 720×480)",
]

NEGATIVE = (
    "worst quality, low quality, blurry, jittery, flicker, inconsistent motion, "
    "deformed anatomy, duplicated limbs, distorted hands, warped face, text, logo, watermark"
)

# Adult mode intentionally does NOT add prudish/censorship terms such as
# "nudity", "adult", "sexual", etc. It only keeps quality/artifact negatives.
ADULT_NEGATIVE = NEGATIVE

# Hard safety exclusions remain even in Adult 18+ mode.
# This is intentionally narrow: the mode is for consensual ADULT content only.
FORBIDDEN_ADULT_TERMS = [
    "minor", "minors", "child", "children", "kid", "kids", "underage",
    "teen", "teenager", "schoolgirl", "schoolboy", "preteen",
    "rape", "raped", "non-consensual", "nonconsensual", "forced sex",
    "sexual assault", "incest", "bestiality", "zoophilia",
]

HUMAN_CUES = [
    "person", "people", "man", "woman", "girl", "boy", "face", "portrait",
    "model", "couple", "body", "hands", "human", "character", "selfie",
    "dancer", "bride", "groom", "mother", "father", "male", "female",
]

HUMAN_POSITIVE_STYLE = (
    "anatomically correct adult human, symmetrical face, balanced eyes, clean skin texture, "
    "natural shoulders, aligned spine, proportional limbs, elegant posture, visually pleasing composition, "
    "natural expression, refined beauty, realistic hands with five fingers on each visible hand, "
    "stable identity, consistent facial features, no uncanny distortion"
)

HUMAN_NEGATIVE = (
    "deformed anatomy, uncanny face, asymmetrical eyes, cross-eyed, uneven pupils, malformed mouth, "
    "crooked jaw, extra fingers, missing fingers, fused fingers, broken wrists, twisted arms, "
    "warped legs, duplicate limbs, duplicate body parts, dislocated joints, contorted posture, "
    "hunched back, floating limbs, body horror, creepy, grotesque, nightmare fuel, mutilation"
)

EXTREME_HUMAN_POSITIVE = (
    "single main subject, premium portrait fidelity, ultra symmetrical face, refined facial aesthetics, "
    "balanced eyes, clean jawline, natural skin, elegant neck and shoulders, realistic clavicles, "
    "natural spine, graceful posture, proportional body, stable identity across the whole shot, "
    "calm elegant motion, hands either fully visible and anatomically correct or naturally out of frame, "
    "five fingers on each visible hand, polished beauty, tasteful cinematography, no uncanny drift"
)

EXTREME_HUMAN_NEGATIVE = (
    "ugly, uncanny valley, frightening, scary face, monster face, malformed nose, broken smile, "
    "wonky eyes, wall-eyed, lazy eye, asymmetrical face, inflated lips, melting face, warped cheeks, "
    "bad teeth, open screaming mouth, distorted ears, twisted neck, bent spine, awkward hips, "
    "short arms, long arms, stubby fingers, spaghetti fingers, claw hands, missing hand, extra hand, "
    "cropped hand near camera, tangled limbs, crowd collision, fisheye distortion, extreme foreshortening"
)

def likely_human_request(text: str, has_image: bool) -> bool:
    lowered = (text or "").lower()
    return has_image or any(word in lowered for word in HUMAN_CUES)

def validate_adult_request(text: str):
    lowered = (text or "").lower()
    hits = [term for term in FORBIDDEN_ADULT_TERMS if term in lowered]
    if hits:
        raise gr.Error(
            "El modo Adulto 18+ solo admite contenido sexual entre adultos que consienten. "
            "La solicitud contiene términos no permitidos para este modo: "
            + ", ".join(sorted(set(hits)))
        )

PRESETS = {
    "Draft 16:9": (576, 320),
    "Cinematic 16:9": (640, 352),
    "Vertical 9:16": (352, 640),
    "Square 1:1": (512, 512),
}

FRAME_PRESETS = {
    "2 seconds": 49,
    "3 seconds": 65,
    "4 seconds": 81,
    "10 seconds (segmentado estable)": 240,
}


def run_cmd(args, timeout=30):
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 1, str(e)


def gpu_info():
    code, txt = run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,temperature.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if code == 0:
        parts = [x.strip() for x in txt.strip().split(",")]
        if len(parts) >= 5:
            return {
                "name": parts[0],
                "total": int(parts[1]),
                "used": int(parts[2]),
                "temp": parts[3],
                "driver": parts[4],
            }
    return {"name": "NVIDIA GPU not detected", "total": 0, "used": 0, "temp": "?", "driver": "?"}


def ollama_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=4)
        r.raise_for_status()
        return True, [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as e:
        return False, [str(e)]


def unload_ollama():
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "keep_alive": 0},
            timeout=20,
        )
    except Exception:
        pass


def direct_prompt_instruction(idea, style, camera, has_image, adult_mode, human_protection, extreme_fidelity):
    image_rule = (
        "- The user also uploaded a reference image that will be used as the FIRST FRAME of the video.\n"
        "- Preserve the same main subject, scene identity, color palette and overall visual continuity.\n"
        "- Write the shot as a continuation/animation of that image, unless the user explicitly asks for a change.\n"
        if has_image else ""
    )
    adult_rule = (
        "- ADULT 18+ MODE is enabled. The scene may contain explicit consensual adult sexuality or nudity if requested.\n"
        "- Do not sanitize, euphemize, moralize, or remove requested adult sexual details.\n"
        "- All depicted people must be clearly adults (18+) and consenting.\n"
        if adult_mode else ""
    )
    human_rule = (
        "- HUMAN FIDELITY MODE is enabled. If humans appear, prioritize beauty, symmetry and clean anatomy.\n"
        "- Prefer one main subject or a very small number of subjects unless the user explicitly asks for more.\n"
        "- Prefer portrait, bust, half-body or otherwise stable framing unless full body is explicitly requested.\n"
        "- Avoid extreme perspective, fisheye distortion, violent foreshortening and crowded tangles of limbs.\n"
        "- Keep hands either clearly visible and fully formed, or naturally out of frame; never ambiguous half-hands near the lens.\n"
        "- Enforce: symmetrical face, balanced eyes, correct limbs, natural posture, elegant pose, five fingers on each visible hand.\n"
        if human_protection else ""
    )
    extreme_rule = (
        "- EXTREME FIDELITY MODE is enabled. Optimize for premium beauty and stability over ambitious motion.\n"
        "- Use one main adult subject unless the user explicitly requires more.\n"
        "- Favor portrait or half-body framing, flattering lensing, gentle camera movement and graceful motion.\n"
        "- Keep the face front-facing or 3/4 view whenever possible, with stable identity and elegant expression.\n"
        "- Keep visible hands relaxed, complete, anatomically correct and away from the lens unless specifically requested.\n"
        "- Avoid choreography, chaotic action, crowding, heavy occlusion, acrobatics, extreme body twists and exaggerated mouth shapes.\n"
        "- Maintain premium facial symmetry, attractive composition, polished aesthetics and non-creepy output.\n"
        if extreme_fidelity else ""
    )
    return f"""
You are the prompt director for a LOCAL video diffusion model.
Rewrite the user's idea as ONE coherent English cinematic shot.

Rules:
- Preserve the user's requested subject and action.
- One continuous scene, not a montage.
{image_rule}{adult_rule}{human_rule}{extreme_rule}- Explicitly describe subject, environment, action, camera movement, lighting,
  atmosphere, textures, depth, motion and temporal consistency.
- Avoid dialogue, subtitles, captions, logos and watermarks unless explicitly requested.
- Aim for physically coherent movement and stable identity.
- 70 to 130 English words.
- Return STRICT JSON only with keys: prompt, title.

User idea:
{idea}

Visual style: {style}
Camera: {camera}
""".strip()


def enhance_with_ollama(idea, style, camera, has_image, adult_mode, human_protection, extreme_fidelity):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": direct_prompt_instruction(idea, style, camera, has_image, adult_mode, human_protection, extreme_fidelity)}],
        "stream": False,
        "format": "json",
        "keep_alive": 0,
        "options": {"temperature": 0.35, "num_ctx": 4096},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    data = json.loads(r.json()["message"]["content"])
    return str(data.get("prompt", idea)).strip(), str(data.get("title", "Local Video")).strip()


def make_fallback_prompt(idea, style, camera, has_image, adult_mode, human_protection, extreme_fidelity):
    prefix = ""
    if has_image:
        prefix = (
            "Animate the uploaded image as the first frame. Preserve the same main subject, scene identity, "
            "color palette and overall continuity. "
        )
    adult_prefix = (
        "All depicted people are consenting adults aged 18 or older. Preserve requested adult sexual or nude content without sanitizing it. "
        if adult_mode else ""
    )
    human_prefix = (
        "If people appear, prioritize beauty and clean anatomy: symmetrical face, balanced eyes, natural shoulders and spine, proportional limbs, elegant posture, stable identity, realistic hands with five fingers on each visible hand, no uncanny or grotesque distortions. Prefer stable portrait or half-body framing unless full-body is explicitly requested. "
        if human_protection else ""
    )
    extreme_prefix = (
        "Optimize for premium portrait fidelity and visual elegance. Use one main adult subject, flattering portrait or half-body framing, calm graceful motion, polished beauty, refined facial aesthetics, ultra symmetrical face, natural expression, elegant posture and complete anatomically correct hands or naturally hidden hands. Avoid crowds, chaotic movement, extreme perspective and distorted mouth or eye shapes. "
        if extreme_fidelity else ""
    )
    return (
        prefix + adult_prefix + human_prefix + extreme_prefix
        + f"{idea.strip()}. {style}. {camera}. One continuous cinematic shot, detailed environment, "
        "natural physically coherent movement, stable subject identity, cinematic lighting, realistic "
        "depth and textures, smooth temporal consistency, no captions, no logos."
    )


def latest_mp4(folder: Path):
    files = sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def generate(engine, idea, style, camera, preset, duration, seed, use_ollama, input_image, image_strength, adult_mode, keep_intermediates, human_protection, extreme_fidelity, progress=gr.Progress()):
    if not idea or not idea.strip():
        raise gr.Error("Escribe primero lo que quieres ver en el video.")

    if adult_mode:
        validate_adult_request(idea)

    premium_engine = str(engine).startswith("👑 CogVideoX")
    if premium_engine and not input_image:
        raise gr.Error(
            "CogVideoX Premium es Image-to-Video: sube una imagen base. "
            "Para Text-to-Video usa LTX FAST."
        )

    if premium_engine:
        required = [
            COG_MODEL_DIR / "transformer" / "config.json",
            COG_MODEL_DIR / "vae" / "config.json",
            COG_MODEL_DIR / "scheduler" / "scheduler_config.json",
        ]
        if not all(p.exists() for p in required):
            raise gr.Error(
                "CogVideoX Premium todavía no está instalado completamente. "
                "Ejecuta INSTALL_COGVIDEOX_PREMIUM_v4_0.bat."
            )

    gpu = gpu_info()
    if gpu["total"] < 11000:
        raise gr.Error(f"Esta edición está ajustada para una RTX 3060 de 12 GB. Detecté {gpu['total']} MiB.")

    model_file = MODELS / "ltxv-2b-0.9.8-distilled.safetensors"
    if not model_file.exists():
        raise gr.Error("Falta el modelo LTX 2B: models/ltxv-2b-0.9.8-distilled.safetensors")

    if int(seed) < 0:
        seed = random.randint(1, 2_147_483_647)
    seed = int(seed)

    has_image = bool(input_image and str(input_image).strip())
    apply_human_protection = bool(human_protection and likely_human_request(idea, has_image))
    apply_extreme_fidelity = bool(extreme_fidelity and apply_human_protection)

    progress(0.04, desc="Preparando prompt")
    title = "Local Video"
    if use_ollama:
        ok, models = ollama_models()
        if ok and OLLAMA_MODEL in models:
            try:
                prompt, title = enhance_with_ollama(idea, style, camera, has_image, adult_mode, apply_human_protection, apply_extreme_fidelity)
            except Exception:
                prompt = make_fallback_prompt(idea, style, camera, has_image, adult_mode, apply_human_protection, apply_extreme_fidelity)
        else:
            prompt = make_fallback_prompt(idea, style, camera, has_image, adult_mode, apply_human_protection, apply_extreme_fidelity)
    else:
        prompt = make_fallback_prompt(idea, style, camera, has_image, adult_mode, apply_human_protection, apply_extreme_fidelity)

    progress(0.10, desc="Liberando Ollama de la GPU")
    unload_ollama()
    time.sleep(1)

    if premium_engine:
        width, height = 720, 480
        frames = 49
        fps = 8
        effective_duration = "6 seconds (CogVideoX native)"
    else:
        width, height = PRESETS[preset]
        frames = FRAME_PRESETS[duration]
        fps = 24
        effective_duration = duration

    job = f"{time.strftime('%Y%m%d_%H%M%S')}_{seed}"
    job_dir = OUTPUTS / job
    job_dir.mkdir(parents=True, exist_ok=True)

    staged_image = None
    if has_image:
        src = Path(str(input_image))
        if not src.exists():
            raise gr.Error("No pude leer la imagen subida.")
        ext = src.suffix if src.suffix else ".png"
        staged_image = job_dir / f"input_image{ext}"
        shutil.copy2(src, staged_image)

    spec = {
        "prompt": prompt,
        "negative_prompt": ((ADULT_NEGATIVE if adult_mode else NEGATIVE) + (", " + HUMAN_NEGATIVE if apply_human_protection else "") + (", " + EXTREME_HUMAN_NEGATIVE if apply_extreme_fidelity else "")),
        "width": width,
        "height": height,
        "num_frames": frames,
        "frame_rate": fps,
        "seed": seed,
        "output_path": str(job_dir),
        "title": title,
        "original_idea": idea,
        "image_path": str(staged_image) if staged_image else None,
        "image_strength": float(image_strength),
        "generation_mode": "image_to_video" if staged_image else "text_to_video",
        "duration_label": effective_duration,
        "engine": "cogvideox" if premium_engine else "ltx",
        "cog_steps": 50 if (premium_engine and apply_extreme_fidelity) else 35,
        "adult_mode": bool(adult_mode),
        "keep_intermediates": bool(keep_intermediates),
        "human_protection": bool(apply_human_protection),
        "extreme_fidelity": bool(apply_extreme_fidelity),
    }
    job_json = RUNTIME / f"{job}.json"
    job_json.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    worker_name = "cog_worker.py" if premium_engine else "worker.py"
    engine_label = "CogVideoX 5B Premium" if premium_engine else "LTX-Video 2B"
    progress(0.16, desc=f"Iniciando {engine_label}")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / worker_name), str(job_json)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    log_lines = []
    stage_map = {
        "[STAGE 1/6]": 0.20,
        "[STAGE 2/6]": 0.32,
        "[STAGE 3/6]": 0.46,
        "[STAGE 4/6]": 0.58,
        "[STAGE 5/6]": 0.93,
        "[STAGE 6/6]": 1.00,
    }
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            clean = line.rstrip()
            log_lines.append(clean)

            handled = False
            for marker, value in stage_map.items():
                if marker in clean:
                    progress(value, desc=clean[-110:])
                    handled = True
                    break

            if "[SEGMENT]" in clean:
                try:
                    fraction = clean.split("[SEGMENT]", 1)[1].strip()
                    done, total = [int(x) for x in fraction.split("/", 1)]
                    value = 0.58 + 0.27 * min(1.0, (done - 1) / max(total, 1))
                    progress(value, desc=f"Segmento {done}/{total}")
                except Exception:
                    pass
                handled = True

            if "[LTX_STEP]" in clean:
                try:
                    fraction = clean.split("[LTX_STEP]", 1)[1].strip()
                    done, total = [int(x) for x in fraction.split("/", 1)]
                    value = 0.58 + 0.30 * min(1.0, done / max(total, 1))
                    progress(value, desc=f"LTX paso {done}/{total}")
                except Exception:
                    progress(0.70, desc=clean[-110:])
                handled = True

            if "[COG_STEP]" in clean:
                try:
                    fraction = clean.split("[COG_STEP]", 1)[1].strip()
                    done, total = [int(x) for x in fraction.split("/", 1)]
                    value = 0.50 + 0.42 * min(1.0, done / max(total, 1))
                    progress(value, desc=f"CogVideoX paso {done}/{total}")
                except Exception:
                    progress(0.70, desc=clean[-110:])
                handled = True

            if not handled and len(log_lines) % 12 == 0:
                progress(0.18, desc=clean[-110:])

        if proc.poll() is not None:
            if proc.stdout:
                rest = proc.stdout.read()
                if rest:
                    log_lines.extend(rest.splitlines())
            break

    if proc.returncode != 0:
        tail = "\n".join(log_lines[-50:])
        if "out of memory" in tail.lower():
            raise gr.Error(
                f"{engine_label} se quedó sin VRAM/memoria. Cierra programas que usen GPU/RAM.\n\n"
                + tail[-2500:]
            )
        raise gr.Error("La generación falló.\n\n" + tail[-3500:])

    out = latest_mp4(job_dir)
    if not out:
        raise gr.Error(f"{engine_label} terminó sin crear un MP4. Revisa el registro de la terminal.")

    metadata = spec | {"video": str(out)}
    (job_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    long_mode = (
        "CogVideoX nativo"
        if premium_engine
        else ("segmentado estable" if "10 seconds" in duration else "single-pass")
    )
    model_label = "CogVideoX-5B-I2V" if premium_engine else "LTX-Video 2B 0.9.8 Distilled"
    progress(1.0, desc="Video listo")
    details = (
        f"### {title}\n"
        f"**Motor:** `{'👑 CogVideoX 5B PREMIUM' if premium_engine else '⚡ LTX 2B FAST'}`  \n"
        f"**Modo:** `{'Image-to-Video' if has_image else 'Text-to-Video'}`  \n"
        f"**Adulto 18+:** `{'ACTIVO' if adult_mode else 'desactivado'}`  \n"
        f"**Protección humana:** `{'ACTIVA' if apply_human_protection else 'desactivada'}`  \n"
        f"**Fidelidad extrema:** `{'ACTIVA' if apply_extreme_fidelity else 'desactivada'}`  \n"
        f"**Duración efectiva:** `{effective_duration}`  \n"
        f"**Motor de duración:** `{long_mode}`  \n"
        f"**Seed:** `{seed}`  \n"
        f"**Resolución:** `{width}×{height}`  \n"
        f"**Frames objetivo:** `{frames}` a {fps} FPS  \n"
        f"**Modelo:** `{model_label}`  \n"
        + (f"**Pasos CogVideoX:** `{spec['cog_steps']}`  \n" if premium_engine else f"**Fuerza de imagen:** `{float(image_strength):.2f}`  \n")
        + f"\n**Prompt dirigido:**\n\n{prompt}"
    )
    return str(out), details, seed



def cleanup_temporaries():
    """
    Limpia únicamente temporales seguros:
    - carpetas outputs/*/segments
    - MP4 atómicos abandonados __writing__*.mp4
    - JSON de runtime que ya no corresponden a una generación activa
    Nunca borra videos finales, metadata.json, input_image.* ni modelos.
    """
    removed_dirs = 0
    removed_files = 0
    reclaimed = 0

    # Si hay un worker activo, no tocar runtime para no interferir.
    worker_active = False
    try:
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or [])
                if (p.info.get("name") or "").lower() == "python.exe" and "worker.py" in cmd and str(ROOT) in cmd:
                    worker_active = True
                    break
            except Exception:
                pass
    except Exception:
        pass

    # Segmentos ya usados por clips largos.
    for d in OUTPUTS.glob("*/segments"):
        if not d.is_dir():
            continue
        try:
            for f in d.rglob("*"):
                if f.is_file():
                    try:
                        reclaimed += f.stat().st_size
                    except Exception:
                        pass
            shutil.rmtree(d)
            removed_dirs += 1
        except Exception:
            pass

    # Archivos de escritura incompleta.
    for f in OUTPUTS.rglob("__writing__*.mp4"):
        try:
            reclaimed += f.stat().st_size
            f.unlink(missing_ok=True)
            removed_files += 1
        except Exception:
            pass

    # JSON de trabajos terminados. Solo si no hay worker activo.
    if not worker_active:
        for f in RUNTIME.glob("*.json"):
            try:
                reclaimed += f.stat().st_size
                f.unlink(missing_ok=True)
                removed_files += 1
            except Exception:
                pass

    mb = reclaimed / (1024 * 1024)
    active_note = " (runtime no se tocó porque hay una generación activa)" if worker_active else ""
    return (
        f"Limpieza terminada: {removed_dirs} carpetas temporales y "
        f"{removed_files} archivos eliminados. Espacio recuperado: {mb:.1f} MB{active_note}"
    )


def diagnostics():
    gpu = gpu_info()
    ok, models = ollama_models()
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT))
    return (
        f"GPU: {gpu['name']}\n"
        f"VRAM: {gpu['used']} / {gpu['total']} MiB\n"
        f"GPU temp: {gpu['temp']} C\n"
        f"NVIDIA driver: {gpu['driver']}\n"
        f"RAM: {vm.total / 1024**3:.1f} GB total | {vm.available / 1024**3:.1f} GB available\n"
        f"Disk free: {disk.free / 1024**3:.1f} GB\n"
        f"Ollama: {'OK' if ok else 'NOT AVAILABLE'}\n"
        f"Ollama model selected: {OLLAMA_MODEL}\n"
        f"Installed Ollama models: {', '.join(models)}\n"
        f"LTX checkpoint: {'OK' if (MODELS / 'ltxv-2b-0.9.8-distilled.safetensors').exists() else 'MISSING'}\n"
        f"CogVideoX Premium: {'OK' if (COG_MODEL_DIR / 'transformer' / 'config.json').exists() else 'NO INSTALADO'}\n"
        f"Modo render: LTX FAST + CogVideoX 5B PREMIUM I2V\n"
        f"Entradas: Text-to-Video + Image-to-Video\n"
        f"T5 INT8: InvokeAI/t5-v1_1-xxl local"
    )


with gr.Blocks(title="Local Video Studio RTX 3060 DUAL ENGINE v4.0 Long Clips") as demo:
    gr.Markdown(
        "# 🎬 Local Video Studio — RTX 3060 DUAL ENGINE v4.0\n"
        "**Prompt → Qwen3:8B → T5 INT8 → LTX-Video 2B SINGLE-PASS → MP4**, ahora con **clips de 10 segundos** y **Modo Adulto 18+** para contenido consensual entre adultos."
    )

    with gr.Tab("Generar"):
        engine = gr.Dropdown(
            ENGINES,
            value=ENGINES[0],
            label="Motor de generación",
            info="LTX = rápido y flexible. CogVideoX Premium = Image-to-Video de mayor calidad, 720×480 y ~6 s.",
        )
        idea = gr.Textbox(
            label="Describe tu video",
            lines=5,
            placeholder="Ejemplo: La persona de la imagen camina lentamente, sonríe y la cámara la sigue durante 10 segundos...",
        )
        input_image = gr.Image(
            type="filepath",
            label="Imagen base (opcional) — si la subes, se usará como primer frame",
        )
        image_strength = gr.Slider(
            minimum=0.70,
            maximum=1.00,
            step=0.05,
            value=0.95,
            label="Fuerza de la imagen",
            info="1.00 = primer frame muy fiel; 0.70 = más libertad de movimiento/interpretación",
        )

        with gr.Row():
            style = gr.Dropdown(
                ["Cinematic photorealism", "Realistic documentary", "Science fiction",
                 "Anime cinematic", "3D animated film", "Luxury commercial", "Dark fantasy"],
                value="Cinematic photorealism",
                label="Estilo",
            )
            camera = gr.Dropdown(
                ["smooth tracking shot", "slow dolly-in", "handheld documentary camera",
                 "cinematic orbit shot", "low-angle tracking shot", "aerial drone shot",
                 "static tripod shot"],
                value="smooth tracking shot",
                label="Cámara",
            )
        with gr.Row():
            preset = gr.Dropdown(list(PRESETS.keys()), value="Draft 16:9", label="Formato / calidad")
            duration = gr.Radio(list(FRAME_PRESETS.keys()), value="2 seconds", label="Duración")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = aleatoria)")

        use_ollama = gr.Checkbox(value=True, label="Director IA con Qwen3:8B (recomendado)")
        adult_mode = gr.Checkbox(
            value=False,
            label="🔞 Modo Adulto 18+ (solo adultos que consienten)",
            info="Evita que el director IA suavice o elimine desnudez/sexualidad adulta solicitada. No permite menores, coerción, incesto ni bestialidad.",
        )
        keep_intermediates = gr.Checkbox(
            value=False,
            label="Conservar archivos intermedios",
            info="Déjalo desactivado para que los segmentos de clips largos y JSON temporales se borren automáticamente tras un render exitoso.",
        )
        human_protection = gr.Checkbox(
            value=True,
            label="✨ Protección humana / estética",
            info="Refuerza simetría facial, manos correctas, postura natural y anatomía limpia cuando hay personas.",
        )
        extreme_fidelity = gr.Checkbox(
            value=True,
            label="👑 Fidelidad extrema / retrato premium",
            info="Prioriza belleza, simetría, manos correctas y estabilidad de identidad por encima de movimientos ambiciosos.",
        )
        generate_btn = gr.Button("🎥 GENERAR VIDEO", variant="primary")
        video = gr.Video(label="Resultado")
        details = gr.Markdown()

        generate_btn.click(
            generate,
            inputs=[engine, idea, style, camera, preset, duration, seed, use_ollama, input_image, image_strength, adult_mode, keep_intermediates, human_protection, extreme_fidelity],
            outputs=[video, details, seed],
        )

    with gr.Tab("Sistema"):
        diag = gr.Textbox(label="Diagnóstico", lines=14)
        with gr.Row():
            refresh = gr.Button("Actualizar diagnóstico")
            clean_btn = gr.Button("🧹 Limpiar temporales")
        clean_status = gr.Textbox(label="Resultado de limpieza", lines=2)
        refresh.click(diagnostics, outputs=diag)
        clean_btn.click(cleanup_temporaries, outputs=clean_status)
        demo.load(diagnostics, outputs=diag)

    gr.Markdown(
        "Los clips de **10 segundos** usan un modo **segmentado estable**: el sistema genera varios bloques cortos, "
        "encadena continuidad con el último frame y luego los une en un MP4 final."
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
    )
