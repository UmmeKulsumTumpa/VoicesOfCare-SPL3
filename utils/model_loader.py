"""
utils/model_loader.py
======================
Centralised model-loading logic used by training, inference, merge, and
evaluation scripts.

Handles multimodal models (e.g. Ministral-3 / Mistral3ForConditionalGeneration)
transparently:
  - Checks the model's config first (tiny download) to determine the
    correct Auto class — avoids downloading weights twice on error.
  - Uses AutoModelForImageTextToText for vision+text models (e.g. mistral3).
  - Uses AutoModelForCausalLM for pure text models.
  - Restricts LoRA targets to language_model.* layers when multimodal,
    so the vision encoder stays fully frozen and uses zero extra VRAM.
"""

from __future__ import annotations

import os
import re
from typing import Optional

# Disable the xet download protocol in huggingface_hub ≥1.4.
# xet does NOT support resuming interrupted downloads; the standard
# HTTP downloader does (via .incomplete files).  This ensures large
# model downloads survive network interruptions.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Transformers 5.x new core_model_loading pipeline materialises tensors
# in bf16 on GPU before BNB quantisation, causing a transient ~6 GB peak
# (fatal on 4 GB GPUs).  expandable_segments reduces fragmentation OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# model_type values that are vision+text (multimodal) and NOT registered
# in MODEL_FOR_CAUSAL_LM_MAPPING.  We load these with the image-text
# auto class instead.
_MULTIMODAL_MODEL_TYPES = {"mistral3", "llava", "paligemma", "idefics2"}


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_bnb_config(cfg: dict, cpu_offload: bool = False) -> BitsAndBytesConfig:
    """Build a BitsAndBytesConfig from the YAML quantization section.

    When *cpu_offload* is True, sets llm_int8_enable_fp32_cpu_offload
    so that modules placed on CPU stay in fp32/bf16 while GPU modules
    are quantized to 4-bit.  Required for multimodal models where the
    vision encoder is offloaded to CPU.
    """
    qcfg = cfg["quantization"]
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_compute_dtype=dtype_map[qcfg["bnb_4bit_compute_dtype"]],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        llm_int8_enable_fp32_cpu_offload=cpu_offload,
    )


def is_multimodal(cfg: dict) -> bool:
    """Check whether the active model is multimodal (text + vision)."""
    return cfg.get("model", {}).get("is_multimodal", False)


def _resolve_auto_class(model_name: str):
    """
    Inspect the model's config.json (tiny download) and return the
    right Auto class so we never download 7 GB of weights only to
    discover the wrong class was used.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    # Check if model_type is registered in the CausalLM mapping
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return AutoModelForCausalLM, "AutoModelForCausalLM"

    # For known multimodal types, use the image-text-to-text class
    if model_type in _MULTIMODAL_MODEL_TYPES:
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText, "AutoModelForImageTextToText"
        except ImportError:
            pass

    # Generic fallback
    return AutoModel, "AutoModel"


def _load_model(
    model_name: str,
    *,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    device_map="auto",
    dtype=torch.bfloat16,
):
    """
    Load the model using the correct Auto class (determined from config).
    Downloads weights exactly once.

    *device_map* can be a string ("auto", "cpu") or a dict that
    explicitly maps top-level modules to devices.
    """
    auto_cls, cls_name = _resolve_auto_class(model_name)

    load_kwargs = dict(
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",
        device_map=device_map,
        # Critical for transformers 5.x on small GPUs (RTX 3050 / 4 GB):
        # Keep weights on CPU during deserialization; BNB quantises each
        # tensor as it is moved to GPU → peak VRAM ~200 MB instead of
        # ~6 GB (full bf16 model).  Without this flag, transformers 5.x
        # core_model_loading.py materialises ALL weights in bf16 on GPU
        # first, then quantises — causing OOM at ~layer 21/36 on 3B models.
        low_cpu_mem_usage=True,
    )
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    model = auto_cls.from_pretrained(model_name, **load_kwargs)
    print(f"  Loaded via {cls_name} ({type(model).__name__})")
    return model


def _build_multimodal_device_map(model_name: str) -> dict:
    """
    Build a fine-grained device_map for multimodal models on small GPUs.

    Strategy for Ministral-3 on RTX 3050 (3.68 GiB):
      - vision_tower          → CPU  (0.60 GB bf16, not used for text)
      - multi_modal_projector → CPU  (0.01 GB bf16, not used for text)
      - embed_tokens          → CPU  (0.81 GB bf16, Embedding — NOT quantized!)
      - 26 decoder layers     → GPU  (1.43 GB in 4-bit)
      - norm, rotary_emb      → GPU  (tiny)
      - lm_head               → GPU  (0.20 GB in 4-bit)
      ─────────────────────────────────
      GPU total: ~1.63 GB — leaves ~2 GB for activations / optimizer

    Accelerate's hooks automatically transfer embed_tokens output
    from CPU → GPU when the first decoder layer needs it.  This adds
    ~1-2 ms per step, which is negligible vs the ~500 ms per step.
    """
    auto_cls, _ = _resolve_auto_class(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    from accelerate import init_empty_weights
    with init_empty_weights():
        empty_model = auto_cls.from_config(config, trust_remote_code=True)

    device_map = {}

    # Walk top-level → depth-2 → depth-3 to build specific mappings
    for top_name, top_module in empty_model.named_children():
        top_children = list(top_module.named_children())

        if not top_children:
            # Top-level leaf (e.g. lm_head) → GPU
            device_map[top_name] = 0
            continue

        for mid_name, mid_module in top_children:
            full_mid = f"{top_name}.{mid_name}"

            # Vision / projector modules → CPU
            if any(kw in mid_name.lower() for kw in ("vision", "multi_modal", "image", "pixel")):
                device_map[full_mid] = "cpu"
                continue

            # Language model sub-modules need deeper mapping
            mid_children = list(mid_module.named_children())
            if not mid_children:
                device_map[full_mid] = 0
                continue

            for leaf_name, leaf_module in mid_children:
                full_leaf = f"{full_mid}.{leaf_name}"

                # embed_tokens is an Embedding (NOT quantized by BNB) and
                # takes 0.81 GB bf16 for 131K vocab — offload to CPU
                if "embed" in leaf_name.lower():
                    device_map[full_leaf] = "cpu"
                elif leaf_name == "layers":
                    # Each decoder layer → GPU
                    for layer_name, _ in leaf_module.named_children():
                        device_map[f"{full_leaf}.{layer_name}"] = 0
                else:
                    # norm, rotary_emb, etc. → GPU (tiny)
                    device_map[full_leaf] = 0

    del empty_model

    # Summarize
    cpu_keys = [k for k, v in device_map.items() if v == "cpu"]
    gpu_keys = [k for k, v in device_map.items() if v == 0]
    print(f"  Device map: {len(gpu_keys)} modules → GPU, {len(cpu_keys)} modules → CPU")
    print(f"  CPU modules: {cpu_keys}")
    return device_map

def _build_low_vram_device_map(model_name: str) -> dict | None:
    """
    Build a device map for text-only CausalLM models on small GPUs (< 5 GB).

    The vocabulary embedding table (embed_tokens) is typically the single
    largest non-quantised tensor.  For Qwen2.5-3B it is 622 MB in bf16
    (151 K vocab \u00d7 2048 hidden \u00d7 2 bytes).  Offloading it to CPU
    frees that headroom for forward-pass allocations (logits, attention,
    cross-entropy workspace) without any quality loss.

    Accelerate installs AlignDevicesHook on CPU-mapped modules:
      • Before forward: weight is moved GPU-side  (PCIe \u2192 ~40 ms)
      • Computation runs on GPU
      • After forward:  weight is moved back to CPU
    So correctness is maintained and only a small per-step latency is added.

    The tied lm_head (same weight tensor as embed_tokens) is also mapped
    to CPU so both reside on the same device as far as accelerate is
    concerned; its forward still executes on GPU via the hook.

    Returns None if VRAM >= 5 GB (offloading unnecessary).
    """
    if not torch.cuda.is_available():
        return None
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_vram_gb >= 5.0:
        return None  # enough VRAM — skip the complexity

    auto_cls, _ = _resolve_auto_class(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    from accelerate import init_empty_weights
    with init_empty_weights():
        empty_model = auto_cls.from_config(config, trust_remote_code=True)

    device_map: dict = {}
    top_children  = dict(empty_model.named_children())

    # lm_head is a direct top-level child in most CausalLM architectures.
    # Its weight is tied to embed_tokens, so it must live on the same device.
    if "lm_head" in top_children:
        device_map["lm_head"] = "cpu"

    # Walk the rest of the model
    for top_name, top_module in top_children.items():
        if top_name == "lm_head":
            continue  # already handled
        for mid_name, mid_module in top_module.named_children():
            full_mid = f"{top_name}.{mid_name}"
            # embed_tokens (Embedding, NOT quantised) → CPU
            if "embed" in mid_name.lower():
                device_map[full_mid] = "cpu"
                continue
            # decoder layers → GPU individually
            if mid_name == "layers":
                for layer_name, _ in mid_module.named_children():
                    device_map[f"{full_mid}.{layer_name}"] = 0
                continue
            # norm, rotary_emb, etc. → GPU (tiny)
            device_map[full_mid] = 0

    del empty_model

    cpu_keys = [k for k, v in device_map.items() if v == "cpu"]
    gpu_count = sum(1 for v in device_map.values() if v == 0)
    print(f"  Low-VRAM device map: {gpu_count} modules \u2192 GPU, {len(cpu_keys)} \u2192 CPU")
    print(f"  CPU-offloaded (frees ~620 MB): {cpu_keys}")
    return device_map
def load_model_and_tokenizer(
    model_name: str,
    cfg: dict,
    *,
    quantize: bool = True,
    device_map: str = "auto",
) -> tuple:
    """
    Load a base model (optionally in 4-bit) and its tokenizer.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.
    cfg : dict
        Full training config (needs 'quantization' section).
    quantize : bool
        If True, load in 4-bit NF4 (training / low-VRAM inference).
        If False, load in bf16 (merging).
    device_map : str
        "auto" for GPU placement, "cpu" for merge scripts.

    Returns
    -------
    (model, tokenizer) tuple
    """
    print(f"\n  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )

    # Many causal LMs do not have a pad token — use EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # For multimodal models: build an explicit device map that puts the
    # vision encoder on CPU and the language model on GPU. This is more
    # reliable than max_memory hints which fail during transient bf16→4bit
    # weight conversion (OOM on small GPUs like RTX 3050 4 GB).
    effective_device_map = device_map
    cpu_offload = False
    if is_multimodal(cfg) and device_map == "auto" and quantize:
        try:
            effective_device_map = _build_multimodal_device_map(model_name)
            cpu_offload = True  # tell BNB that CPU modules are intentional
        except Exception as e:
            print(f"  Warning: could not build explicit device map ({e}), falling back to 'auto'")
            effective_device_map = "auto"
    elif not is_multimodal(cfg) and device_map == "auto" and quantize:
        # For text-only quantized models on small GPUs: build an explicit
        # device map that offloads embed_tokens/lm_head to CPU so BNB is
        # aware of it and won't crash with "make sure you have enough GPU RAM".
        # Without this, device_map="auto" silently places some layers on CPU
        # but BNB 4-bit requires llm_int8_enable_fp32_cpu_offload=True to
        # allow any CPU-resident modules.
        try:
            low_vram_map = _build_low_vram_device_map(model_name)
            if low_vram_map is not None:
                effective_device_map = low_vram_map
                cpu_offload = True
        except Exception as e:
            print(f"  Warning: could not build low-VRAM device map ({e}), falling back to 'auto'")
            effective_device_map = "auto"
    bnb = build_bnb_config(cfg, cpu_offload=cpu_offload) if quantize else None
    dtype = torch.bfloat16

    print(f"  Loading model: {model_name}  "
          f"({'4-bit QLoRA' if quantize else 'bf16'})")
    model = _load_model(
        model_name,
        quantization_config=bnb,
        device_map=effective_device_map,
        dtype=dtype,
    )

    return model, tokenizer


def get_lora_target_modules(cfg: dict) -> list[str]:
    """
    Return the LoRA target modules for the chosen architecture.

    For multimodal models the short names (q_proj, etc.) are returned
    as-is — PEFT's ``modules_to_save`` filtering is handled by passing
    ``target_modules`` together with ``modules_to_save=None``.

    If the config lists a regex-style string instead of a list, it is
    returned as a single-element list (PEFT supports regex patterns).
    """
    arch = cfg["model"]["architecture"]
    targets_map = cfg["model"]["lora_targets"]
    if arch not in targets_map:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Valid options: {list(targets_map.keys())}"
        )
    return targets_map[arch]
