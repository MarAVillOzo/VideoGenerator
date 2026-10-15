from __future__ import annotations

import gc
import warnings
from pathlib import Path

import accelerate
import bitsandbytes as bnb
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import T5Config, T5EncoderModel, T5Tokenizer

warnings.filterwarnings(
    "ignore",
    message=r".*MatMul8bitLt: inputs will be cast.*",
)

INT8_REPO = "InvokeAI/t5-v1_1-xxl"


class InvokeInt8Params(bnb.nn.Int8Params):
    """Int8Params that preserves already-quantized LLM.int8 weights when moved to CUDA."""

    def cuda(self, device=None):
        if self.has_fp16_weights:
            return super().cuda(device)
        if self.CB is not None and self.SCB is not None:
            self.data = self.data.cuda(device)
            self.CB = self.data
            self.SCB = self.SCB.cuda(device)
        else:
            B = self.data.contiguous().half().cuda(device)
            CB, SCB, _ = bnb.functional.int8_vectorwise_quant(B)
            self.data = CB
            self.CB = CB
            self.SCB = SCB
        return self


class InvokeLinear8bitLt(bnb.nn.Linear8bitLt):
    """Loader-compatible Linear8bitLt for InvokeAI's pre-quantized T5 checkpoint."""

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight = state_dict.pop(prefix + "weight")
        bias = state_dict.pop(prefix + "bias", None)
        scb = state_dict.pop(prefix + "SCB", None)
        weight_format = state_dict.pop(prefix + "weight_format", None)

        if weight_format is not None:
            # InvokeAI's published checkpoint currently uses format 0.
            if torch.is_tensor(weight_format):
                weight_format = int(weight_format.item())
            if weight_format != 0:
                raise RuntimeError(f"Unsupported bitsandbytes weight_format={weight_format}")

        if scb is not None:
            self.weight = InvokeInt8Params(
                data=weight,
                requires_grad=False,
                has_fp16_weights=False,
                CB=weight,
                SCB=scb,
            )
            self.bias = bias if bias is None else torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.weight = InvokeInt8Params(
                data=weight,
                requires_grad=False,
                has_fp16_weights=False,
                CB=None,
                SCB=None,
            )
            self.bias = bias if bias is None else torch.nn.Parameter(bias, requires_grad=False)

        new_state = bnb.MatmulLtState()
        new_state.threshold = self.state.threshold
        new_state.has_fp16_weights = False
        new_state.use_pool = self.state.use_pool
        self.state = new_state

    def forward(self, x: torch.Tensor):
        matmul_state = bnb.MatmulLtState()
        matmul_state.threshold = self.state.threshold
        matmul_state.has_fp16_weights = self.state.has_fp16_weights
        matmul_state.use_pool = self.state.use_pool
        matmul_state.is_training = self.training

        if self.weight.CB is None:
            raise RuntimeError("INT8 weight is not initialized/quantized.")
        matmul_state.CB = self.weight.CB
        matmul_state.SCB = self.weight.SCB

        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        return bnb.matmul(x, self.weight, bias=self.bias, state=matmul_state)


def _convert_linear_layers_to_llm_8bit(
    module: torch.nn.Module,
    ignore_modules: set[str],
    outlier_threshold: float = 6.0,
    prefix: str = "",
):
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(
            fullname.startswith(s) for s in ignore_modules
        ):
            replacement = InvokeLinear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
                threshold=outlier_threshold,
            )
            replacement.weight.data = child.weight.data
            if child.bias is not None:
                replacement.bias.data = child.bias.data
            replacement.requires_grad_(False)
            module.__setattr__(name, replacement)
        else:
            _convert_linear_layers_to_llm_8bit(
                child, ignore_modules, outlier_threshold, fullname
            )


def quantize_model_llm_int8(model: torch.nn.Module):
    _convert_linear_layers_to_llm_8bit(model, set())
    return model


def resolve_int8_root() -> Path:
    """Resolve the already-downloaded InvokeAI INT8 snapshot without downloading anything."""
    snapshot = snapshot_download(
        INT8_REPO,
        allow_patterns=["bnb_llm_int8/*"],
        local_files_only=True,
    )
    root = Path(snapshot) / "bnb_llm_int8"
    te = root / "text_encoder_2"
    tok = root / "tokenizer_2"
    weights = te / "bnb_llm_int8_model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(
            "No encuentro bnb_llm_int8_model.safetensors. "
            "La descarga INT8 no está completa."
        )
    if not (te / "config.json").exists() or not tok.exists():
        raise FileNotFoundError("Faltan config/tokenizer del T5 INT8.")
    return root


def load_t5_int8_to_cuda():
    root = resolve_int8_root()
    te_path = root / "text_encoder_2"
    tok_path = root / "tokenizer_2"

    tokenizer = T5Tokenizer.from_pretrained(
        tok_path,
        max_length=512,
        local_files_only=True,
    )
    config = T5Config.from_pretrained(te_path, local_files_only=True)

    # Critical: create both the T5 and replacement int8 linears on meta device.
    # This avoids ever allocating the original ~19 GB BF16 T5.
    with accelerate.init_empty_weights():
        model = T5EncoderModel(config)
        model = quantize_model_llm_int8(model)

    state_dict = load_file(str(te_path / "bnb_llm_int8_model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    allowed_missing = {"encoder.embed_tokens.weight"}
    unexpected = set(unexpected)
    missing = set(missing)
    bad_missing = missing - allowed_missing
    if unexpected or bad_missing:
        raise RuntimeError(
            f"Carga INT8 incompleta. missing={sorted(bad_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    model.tie_weights()
    model.eval()

    # InvokeInt8Params.cuda() knows how to transfer the already-quantized weights.
    model = model.cuda()
    return tokenizer, model


@torch.inference_mode()
def encode_pair(prompt: str, negative_prompt: str, max_length: int = 256):
    tokenizer, model = load_t5_int8_to_cuda()

    def encode(text: str):
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()

        hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        hidden_cpu = hidden.to(device="cpu", dtype=torch.bfloat16)
        mask_cpu = attention_mask.to("cpu")

        del hidden, input_ids, attention_mask, tokens
        return hidden_cpu, mask_cpu

    prompt_embeds, prompt_mask = encode(prompt)
    negative_embeds, negative_mask = encode(negative_prompt or "")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

    return prompt_embeds, prompt_mask, negative_embeds, negative_mask
