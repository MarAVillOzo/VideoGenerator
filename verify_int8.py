import gc
import sys
import torch

from int8_loader import resolve_int8_root, encode_pair

print("=== VERIFICACION T5 INT8 ===", flush=True)
print("CUDA:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA no disponible.")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("INT8 ROOT:", resolve_int8_root(), flush=True)
print("Cargando T5 INT8 y calculando embeddings...", flush=True)

p, pm, n, nm = encode_pair(
    "cinematic woman dancing in Ibiza at sunset, smooth camera movement",
    "blurry, low quality",
    max_length=256,
)

print("PROMPT EMBEDS:", tuple(p.shape), p.dtype, flush=True)
print("NEGATIVE EMBEDS:", tuple(n.shape), n.dtype, flush=True)
print("VRAM DESPUES DE LIBERAR T5: %.2f GB" % (torch.cuda.memory_allocated() / 1024**3), flush=True)
print("INT8_ENCODER_OK", flush=True)
