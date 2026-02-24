#!/usr/bin/env python3
"""
scripts/03_merge_adapter.py
============================
Merge the trained LoRA adapter weights into the base model weights,
producing a single standalone model that can be loaded without PEFT.

Why merge?
  - Merging eliminates the LoRA overhead at inference time.
  - The merged model can be shared on Hugging Face Hub or
    served via llama.cpp / Ollama.
  - The merged model is saved in bfloat16 (not 4-bit) so it is
    ~2× larger on disk but faster to reload for inference.

Usage:
  cd training/
  python scripts/03_merge_adapter.py [--config config/training_config.yaml]
                                      [--adapter_path ./output/run_01]
                                      [--merged_path  ./output/merged]

Outputs:
  output/merged/
      config.json
      model.safetensors   (or shards if model is large)
      tokenizer/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.model_loader import _resolve_auto_class


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    adapter_path = args.adapter_path or cfg["inference"]["adapter_path"]
    merged_path = args.merged_path or cfg["inference"]["merged_path"]
    model_name = cfg["model"]["name"]

    print("\n" + "=" * 65)
    print("  Merging LoRA Adapter into Base Model")
    print("=" * 65)
    print(f"  Base model   : {model_name}")
    print(f"  Adapter path : {adapter_path}")
    print(f"  Merged output: {merged_path}")
    print("=" * 65 + "\n")

    if not Path(adapter_path).exists():
        sys.exit(
            f"ERROR: Adapter not found at '{adapter_path}'.\n"
            "Run  python scripts/02_train_qlora.py  first."
        )

    # ── Load base model in bfloat16 (NOT 4-bit for merging) ──────────────────
    print("Loading base model in bf16 for merging (this uses more RAM)...")
    auto_cls, cls_name = _resolve_auto_class(model_name)
    base_model = auto_cls.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cpu",         # merge on CPU to avoid VRAM exhaustion
        trust_remote_code=True,
    )
    print(f"  Loaded via {cls_name}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    # ── Apply adapter & merge ─────────────────────────────────────────────────
    print(f"Loading LoRA adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging LoRA weights into base model (may take 1-2 minutes)...")
    merged_model = peft_model.merge_and_unload()

    # ── Save merged model ─────────────────────────────────────────────────────
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    print(f"\nSaving merged model → {merged_path}")
    merged_model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    print("\n" + "─" * 65)
    print("Merge complete!")
    print(f"Merged model saved to: {merged_path}")
    print("\nYou can now run inference with:")
    print(f"  python scripts/04_chat_inference.py --use_merged")
    print("─" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into the base model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to saved LoRA adapter directory",
    )
    parser.add_argument(
        "--merged_path",
        type=str,
        default=None,
        help="Output path for the merged model",
    )

    args = parser.parse_args()
    main(args)
