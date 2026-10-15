from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

REPO = "Lightricks/LTX-Video"
FILES = [
    "ltxv-2b-0.9.8-distilled.safetensors",
    "ltxv-spatial-upscaler-0.9.8.safetensors",
]

print("Downloading official LTX-Video files...")
for filename in FILES:
    target = MODELS / filename
    if target.exists() and target.stat().st_size > 100_000_000:
        print(f"[OK] {filename} already exists")
        continue
    print(f"[DOWNLOAD] {filename}")
    path = hf_hub_download(repo_id=REPO, filename=filename, repo_type="model", local_dir=str(MODELS))
    print(f"[OK] {path}")
print("Models ready.")
