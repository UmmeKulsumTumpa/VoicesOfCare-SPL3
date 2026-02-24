#!/usr/bin/env python3
"""
scripts/02_train_qlora.py
==========================
QLoRA fine-tuning of a small language model (SLM) on the
Alzheimer/Dementia caregiver chatbot dataset.

Strategy Overview:
  - Base model loaded in 4-bit NF4 quantization (bitsandbytes)
  - LoRA adapters injected via PEFT (only adapters are trained)
  - SFTTrainer (TRL) orchestrates the training loop
  - Gradient checkpointing + paged AdamW to minimize VRAM usage
  - Designed to run on a 4 GB RTX 3050 Mobile

Memory budget estimate (Ministral-3-3B, 3.4B text + 0.4B vision):
  ┌─────────────────────────────────────┬──────────────┐
  │ Component                           │ VRAM         │
  ├─────────────────────────────────────┼──────────────┤
  │ Model weights (4-bit NF4 + DQ)      │ ~2.0 GB      │
  │ LoRA adapters (r=16, text LM only)  │ ~0.2 GB      │
  │ Activations (seq=1024, batch=1)     │ ~0.6 GB      │
  │ Optimizer states (paged AdamW 8bit) │ ~0.2 GB      │
  ├─────────────────────────────────────┼──────────────┤
  │ Total                               │ ~3.0 GB      │
  └─────────────────────────────────────┴──────────────┘

Usage:
  cd training/
  python scripts/02_train_qlora.py [--config config/training_config.yaml]
                                    [--model mistralai/Ministral-3-3B-Instruct-2512-BF16]
                                    [--output_dir ./output/run_02]
                                    [--max_steps 500]   # quick smoke-test

Outputs:
  output/run_01/
      adapter_config.json
      adapter_model.safetensors
      tokenizer/
      trainer_state.json
      checkpoint-*/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Suppress HuggingFace tokenizers fork warning when dataloader_num_workers > 0
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Reduce CUDA memory fragmentation — critical on low-VRAM GPUs (RTX 3050 4 GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml

# ── Allow imports from project root ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.prompt_utils import format_prompt_for_training
from utils.model_loader import (
    build_bnb_config,
    load_model_and_tokenizer,
    get_lora_target_modules,
)


# ─────────────────────────────────────────────────────────────────────────────
# VRAM-critical imports come last so errors are clear
# ─────────────────────────────────────────────────────────────────────────────
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_lora_config(cfg: dict, target_modules: list[str]) -> LoraConfig:
    lcfg = cfg["lora"]
    return LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
    )


class TrainingProgressCallback(TrainerCallback):
    """Prints a human-readable progress banner at each logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            total = state.max_steps
            loss = logs.get("loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            if isinstance(lr, float):
                lr = f"{lr:.2e}"
            print(
                f"  step {step:>5}/{total}  |  loss: {loss}  |  lr: {lr}"
                + (f"  |  val_loss: {logs.get('eval_loss', '')}" if "eval_loss" in logs else "")
            )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset formatting
# ─────────────────────────────────────────────────────────────────────────────

def make_formatting_func(eos_token: str):
    """
    Return a formatting function compatible with SFTTrainer's
    `formatting_func` parameter.  Each call receives a dict of lists
    (batched) and returns a list of formatted strings.
    """

    def formatting_func(example: dict) -> list[str]:
        texts = []
        for instr, inp, out in zip(
            example["instruction"],
            example["input"],
            example["output"],
        ):
            texts.append(
                format_prompt_for_training(
                    {"instruction": instr, "input": inp, "output": out},
                    eos_token=eos_token,
                )
            )
        return texts

    return formatting_func


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── Config ───────────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # Override config values with CLI args
    model_name = args.model or cfg["model"]["name"]
    output_dir = args.output_dir or cfg["training"]["output_dir"]
    max_steps = args.max_steps  # None means: use num_train_epochs

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_file = str(processed_dir / "train.jsonl")
    val_file = str(processed_dir / "val.jsonl")

    print("\n" + "=" * 65)
    print("  Alzheimer/Dementia Caregiver Chatbot — QLoRA Training")
    print("=" * 65)
    print(f"  Model       : {model_name}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Train file  : {train_file}")
    print(f"  Val file    : {val_file}")
    print(f"  Device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 65 + "\n")

    if not Path(train_file).exists():
        sys.exit(
            f"ERROR: {train_file} not found.\n"
            "Run  python scripts/01_prepare_dataset.py  first."
        )

    # ── Build model / tokenizer ───────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        model_name, cfg, quantize=True, device_map="auto",
    )

    # Prepare for k-bit training (freeze base, enable grad for adapters)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
    )

    # ── CRITICAL: recast lm_head + embed_tokens to bf16 ──────────────────────
    # prepare_model_for_kbit_training() upcasts all non-quantized weights
    # (including lm_head and embed_tokens, which are TIED) to fp32.
    # That turns the 594 MiB bf16 tensor into a 1187 MiB fp32 tensor and
    # leaves only ~308 MiB free — not enough for any lm_head matmul.
    # Casting back to bf16 reclaims ~765 MiB and leaves ~1500 MiB free.
    # lm_head.weight and embed_tokens.weight are the same physical tensor
    # (weight tying), so one .data assignment covers both.
    _lm_weight = model.lm_head.weight
    if _lm_weight.dtype == torch.float32:
        model.lm_head.weight.data = _lm_weight.data.to(torch.bfloat16)
        # embed_tokens is tied — update its reference too so it stays consistent
        model.model.embed_tokens.weight = model.lm_head.weight
        torch.cuda.empty_cache()
        free_mb = torch.cuda.mem_get_info()[0] / 1024**2
        print(f"  lm_head recasted fp32→bf16, saving 594 MiB  |  GPU free: {free_mb:.0f} MiB")

    target_modules = get_lora_target_modules(cfg)
    print(f"\nLoRA target modules: {target_modules}")
    lora_cfg = build_lora_config(cfg, target_modules)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    raw = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file},
        split={"train": "train", "validation": "validation"},
    )
    print(f"  Train: {len(raw['train']):,} examples")
    print(f"  Val  : {len(raw['validation']):,} examples")

    # Pre-format each example into a single "text" string.
    # TRL 0.15+ requires formatting_func OR a text column — not both.
    # Doing the formatting here (map) lets us keep completion_only_loss=True
    # by telling SFTConfig exactly where the response starts.
    eos = tokenizer.eos_token

    def _format_row(example: dict) -> dict:
        return {
            "text": format_prompt_for_training(example, eos_token=eos)
        }

    print("  Pre-formatting dataset...")
    raw["train"] = raw["train"].map(_format_row, num_proc=1)
    raw["validation"] = raw["validation"].map(_format_row, num_proc=1)
    print("  Pre-formatting done.")

    # ── Training args ─────────────────────────────────────────────────────────
    tcfg = cfg["training"]

    # Build SFTConfig (subclass of TrainingArguments)
    sft_cfg = SFTConfig(
        output_dir=output_dir,

        # Sequence length
        max_length=tcfg["max_seq_length"],
        # Dataset already has a "text" field from our map() above.
        dataset_text_field="text",

        # Train on the full formatted sequence (prompt + response).
        # completion_only_loss=True requires response_template wired through
        # DataCollatorForCompletionOnlyLM, which is incompatible with
        # formatting_func in TRL 0.28+.  Full-sequence training is standard
        # and produces equivalent quality for instruction fine-tuning.
        completion_only_loss=False,

        # Epochs / steps
        num_train_epochs=tcfg["num_train_epochs"] if max_steps is None else 1,
        max_steps=max_steps if max_steps else -1,

        # Batch & accumulation
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],

        # Optimizer & LR
        optim=tcfg["optim"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],

        # Mixed precision
        bf16=tcfg["bf16"],
        fp16=tcfg["fp16"],

        # Memory
        gradient_checkpointing=tcfg["gradient_checkpointing"],

        # Packing (set False to keep per-example loss boundaries clear)
        packing=tcfg["packing"],

        # Liger Kernel: fused chunked cross-entropy — avoids materialising
        # the full (B×S×vocab) logits tensor.  Saves ~290 MB peak VRAM.
        use_liger_kernel=tcfg.get("use_liger_kernel", False),

        # Logging & saving
        logging_steps=tcfg["logging_steps"],
        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg["eval_steps"],
        save_strategy=tcfg["save_strategy"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=tcfg["greater_is_better"],

        # Reproducibility
        seed=cfg["data"]["seed"],
        data_seed=cfg["data"]["seed"],

        # Reporting
        report_to=tcfg["report_to"],
        run_name=tcfg["run_name"],

        # Parallel data loading — speeds up CPU-bound tokenization/loading
        dataloader_num_workers=2,
    )

    # ── Build SFTTrainer ──────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=raw["train"],
        eval_dataset=raw["validation"],
        args=sft_cfg,
        callbacks=[TrainingProgressCallback()],
    )

    # ── Print parameter count ─────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\nParameters — total: {total_params:,}  |  "
        f"trainable (LoRA): {trainable_params:,}  |  "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    # Flush PyTorch's GPU memory pool before training.
    # Failed/interrupted runs leave "reserved but unallocated" blocks in the
    # caching allocator (~80 MB on RTX 3050) that won't be given to new
    # allocations until explicitly released.  This frees ~80 MB, giving
    # Liger's 594 MB CE chunk room to fit within the available ~628 MB.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
    print(f"  GPU free before training: {free_mb:.0f} MB")

    print("\nStarting training...\n")
    resume = args.resume_from_checkpoint
    if resume:
        print(f"  Resuming from checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume or None)

    # ── Save adapter ──────────────────────────────────────────────────────────
    print(f"\nSaving LoRA adapter → {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 65)
    print("  Training complete!")
    print(f"  Adapter saved to: {output_dir}")
    print("  Next step: python scripts/03_merge_adapter.py")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune a causal LM on the caregiver dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to YAML config (default: config/training_config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name (e.g. microsoft/phi-2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory for saving the adapter",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps (useful for quick smoke-test, e.g. 50)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test: run 50 steps only",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint dir to resume training from "
             "(e.g. ./output/run_01/checkpoint-2500). "
             "Restores optimizer, scheduler, and RNG state.",
    )

    args = parser.parse_args()
    if args.smoke and args.max_steps is None:
        args.max_steps = 50
    main(args)
