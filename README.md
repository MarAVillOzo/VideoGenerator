# Local Video Studio — RTX 3060 12 GB Edition V2

Optimized for your RTX 3060 12 GB and local Ollama installation.

## What you can do immediately

Write a prompt in Spanish or English and generate an MP4 locally.
With Director IA enabled, qwen3:8b rewrites your idea as a cinematic prompt,
then Ollama is unloaded and LTX-Video renders the clip.

## Install

1. Extract the ZIP.
2. Double-click `INSTALL.bat`.
3. When complete, double-click `DOWNLOAD_MODELS.bat`.
   It downloads about 6.9 GB of official LTX model files.
4. Double-click `START.bat`.
5. The interface opens in your browser.

The first actual generation can additionally download the PixArt/T5 text encoder
used by official LTX inference, so keep plenty of disk space free.

## Recommended first generation

- Draft 16:9
- 2 seconds
- Director IA: ON

Example:
"Un astronauta camina por una ciudad futurista bajo una lluvia intensa,
con luces de neón reflejadas en el pavimento."

## Memory strategy

- LTX-Video 2B 0.9.8 Distilled
- qwen3:8b for prompt direction
- Ollama keep_alive=0
- official CPU offload enabled
- LTX extra LLM prompt enhancer disabled
- each video render runs in a separate process to release GPU memory cleanly

## Output

Each job is saved under `outputs/` with MP4 and metadata JSON.

## Important

This is real local generative video inference. It gives you a Sora/Grok-style
workflow, but it does not claim cloud-model parity: those services use much larger
models and accelerators.
