#!/usr/bin/env python3
"""
scripts/04_chat_inference.py
=============================
Interactive chatbot for Alzheimer/Dementia caregivers.

Two modes:
  1. Structured mode  — walks the user through filling in patient
     details (age, gender, symptoms, etc.) then generates a response.
  2. Free-text mode   — accepts a raw caregiver post and generates
     a supportive response directly.

Loads either:
  a) The LoRA adapter on top of the quantized base model (default, lower VRAM)
  b) The fully merged model (faster, use --use_merged)

Usage:
  cd training/
  python scripts/04_chat_inference.py [--config config/training_config.yaml]
                                       [--use_merged]
                                       [--mode structured|freetext]
                                       [--adapter_path ./output/run_01]
                                       [--merged_path  ./output/merged]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.prompt_utils import build_inference_prompt, extract_response
from utils.model_loader import load_model_and_tokenizer as _load_base


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_adapter_model(model_name: str, adapter_path: str, cfg: dict):
    """Load 4-bit base model + LoRA adapter (low VRAM mode)."""
    from peft import PeftModel

    model, tokenizer = _load_base(model_name, cfg, quantize=True, device_map="auto")

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Reload tokenizer from adapter dir (may have special tokens saved)
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        pass  # keep the base tokenizer

    return model, tokenizer


def load_merged_model(merged_path: str, cfg: dict):
    """Load the fully merged model directly.

    On GPUs with <8 GB VRAM the bf16 merged model (~6 GB) won't fit,
    so we load on CPU.  Generation is slower but fully functional.
    Use the adapter mode (--no use_merged) for GPU-accelerated inference.
    """
    free_vram = 0
    if torch.cuda.is_available():
        free_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # Need ~6.5 GB free VRAM for bf16 Qwen2.5-3B; fall back to CPU otherwise
    device_map = "auto" if free_vram >= 6.5 else "cpu"
    if device_map == "cpu":
        print("  [merged] VRAM insufficient for bf16 — loading on CPU (slower but functional)")

    model, tokenizer = _load_base(merged_path, cfg, quantize=False, device_map=device_map)
    model.eval()
    return model, tokenizer


def _infer_device(model) -> torch.device:
    """Return the first real (non-meta) device found in model parameters."""
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Chat template & generation
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a compassionate and knowledgeable support assistant for caregivers "
    "of Alzheimer's and dementia patients. Your role is to provide empathetic, "
    "practical, and supportive advice. Acknowledge the caregiver's feelings first, "
    "then offer concrete suggestions. Keep responses warm, clear, and respectful."
)


def build_chat_prompt(tokenizer, user_message: str) -> str:
    """Format a message using Qwen2.5's native ChatML template.

    Uses <|im_start|>system/user/assistant<|im_end|> — the format Qwen2.5-Instruct
    was originally trained on.  This produces far better instruction-following than
    the Alpaca template (### Instruction / ### Response) used during fine-tuning.
    The LoRA adapters still contribute domain vocabulary and tone on top.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # appends <|im_start|>assistant\n
    )


def generate_response(
    model,
    tokenizer,
    prompt: str,
    gen_cfg: dict,
    device: str = "cuda",
) -> str:
    """Run inference and return the model's response text.

    prompt must already be formatted via build_chat_prompt() so Qwen2.5's
    native <|im_start|>/<|im_end|> tokens are present.
    """
    target_device = _infer_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900,   # leave room for generation tokens
    ).to(target_device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            top_k=gen_cfg["top_k"],
            repetition_penalty=gen_cfg["repetition_penalty"],
            do_sample=gen_cfg["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Interaction modes
# ─────────────────────────────────────────────────────────────────────────────

def structured_chat(model, tokenizer, gen_cfg: dict) -> None:
    """Walk user through patient details, then reply."""
    print("\n" + "═" * 60)
    print("  Alzheimer / Dementia Caregiver Support Chatbot")
    print("  (Structured Mode — Press Ctrl+C to exit)")
    print("═" * 60)

    while True:
        try:
            print("\n── New conversation ──────────────────────────────────")
            age = input("Patient age (or 'unknown'): ").strip() or "unknown"
            gender = input("Patient gender (male/female/unknown): ").strip() or "unknown"

            sym_input = input("Known symptoms (comma-separated, or leave blank): ").strip()
            known_symptoms = [s.strip() for s in sym_input.split(",") if s.strip()]

            rare_input = input("Rare symptoms (comma-separated, or leave blank): ").strip()
            rare_symptoms = [s.strip() for s in rare_input.split(",") if s.strip()]

            motivation = input("Why are you posting? (e.g. Seeking_Emotional_Support): ").strip()
            motivations = [motivation] if motivation else []

            relationship = input("Your relationship to the patient: ").strip() or "unspecified"

            print("\nPaste or type your message (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            post_text = "\n".join(lines).strip()

            if not post_text:
                print("[No message entered. Skipping.]")
                continue

            print("\n[Generating response...]")
            user_message = (
                f"Patient Age: {age}\n"
                f"Gender: {gender}\n\n"
                f"Known Symptoms: {known_symptoms}\n"
                f"Rare Symptoms: {rare_symptoms}\n\n"
                f"Caregiver Motivation: {motivations}\n"
                f"Caregiver Relationship: {relationship}\n\n"
                f"{post_text.strip()}"
            )
            prompt = build_chat_prompt(tokenizer, user_message)

            response = generate_response(model, tokenizer, prompt, gen_cfg)
            print("\n" + "─" * 60)
            print("CHATBOT RESPONSE:")
            print("─" * 60)
            print(response)
            print("─" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye. Take care of yourself too. 💙")
            break


def freetext_chat(model, tokenizer, gen_cfg: dict) -> None:
    """Accept raw post text and reply."""
    print("\n" + "═" * 60)
    print("  Alzheimer / Dementia Caregiver Support Chatbot")
    print("  (Free-text Mode — type 'quit' to exit)")
    print("═" * 60)

    while True:
        try:
            print("\nYou (describe your situation):")
            lines = []
            while True:
                line = input()
                if line.strip().lower() == "quit":
                    raise KeyboardInterrupt
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            user_text = "\n".join(lines).strip()

            if not user_text:
                continue

            # Use Qwen2.5's native chat template for best instruction-following
            prompt = build_chat_prompt(tokenizer, user_text)

            print("\n[Generating response...]")
            response = generate_response(model, tokenizer, prompt, gen_cfg)
            print("\n" + "─" * 60)
            print("CHATBOT:")
            print(response)
            print("─" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye. Take care of yourself too. 💙")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    inf_cfg = cfg["inference"]
    gen_cfg = inf_cfg.copy()

    use_merged = args.use_merged or inf_cfg.get("use_merged", False)
    adapter_path = args.adapter_path or inf_cfg["adapter_path"]
    merged_path = args.merged_path or inf_cfg["merged_path"]

    if use_merged:
        if not Path(merged_path).exists():
            sys.exit(
                f"ERROR: Merged model not found at '{merged_path}'.\n"
                "Run  python scripts/03_merge_adapter.py  first, or omit --use_merged."
            )
        model, tokenizer = load_merged_model(merged_path, cfg)
    else:
        if not Path(adapter_path).exists():
            sys.exit(
                f"ERROR: Adapter not found at '{adapter_path}'.\n"
                "Run  python scripts/02_train_qlora.py  first."
            )
        model, tokenizer = load_adapter_model(cfg["model"]["name"], adapter_path, cfg)

    mode = args.mode or "freetext"
    if mode == "structured":
        structured_chat(model, tokenizer, gen_cfg)
    else:
        freetext_chat(model, tokenizer, gen_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive caregiver chatbot inference."
    )
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument(
        "--use_merged",
        action="store_true",
        help="Load merged model instead of adapter + base model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["structured", "freetext"],
        default="freetext",
        help="Interaction mode (default: freetext)",
    )
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--merged_path", type=str, default=None)

    args = parser.parse_args()
    main(args)
