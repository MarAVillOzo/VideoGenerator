from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "ltxv-2b-0.9.8-distilled.safetensors"

print("=== LOCAL VIDEO STUDIO STABLE v3.0 / PRECHECK ===")
print("Python:", sys.version.split()[0])
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM_GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
print("LTX_CHECKPOINT:", "OK" if CHECKPOINT.exists() else "MISSING")

from int8_loader import resolve_int8_root
print("T5_INT8:", resolve_int8_root())
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
print("LTX_IMPORT: OK")

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA no disponible")
if not CHECKPOINT.exists():
    raise SystemExit("ERROR: falta el checkpoint LTX 2B")
print("PRECHECK_OK")
