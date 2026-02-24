#!/usr/bin/env python3
"""
scripts/05_evaluate.py
=======================
Evaluate the fine-tuned model on the held-out test set using:
  - ROUGE-1 / ROUGE-2 / ROUGE-L  (n-gram overlap)
  - BERTScore F1                  (semantic similarity)
  - Perplexity                    (model confidence on gold outputs)

These metrics together give a multi-dimensional view of quality:
  ROUGE   → lexical fidelity to reference responses
  BERTScore → meaning-level similarity (more important for empathetic text)
  Perplexity → how well the model models the gold-standard distribution

Usage:
  cd training/
  python scripts/05_evaluate.py [--config config/training_config.yaml]
                                  [--adapter_path ./output/run_01]
                                  [--use_merged]
                                  [--max_eval_samples 200]

Outputs:
  output/predictions.jsonl   — model predictions on test set
  output/eval_results.json   — aggregated metric scores
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Reduce CUDA memory fragmentation (helps on low-VRAM GPUs)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.prompt_utils import format_prompt_for_inference, format_prompt_for_training
from utils.model_loader import load_model_and_tokenizer as _load_base


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Load model (uses shared loader)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg: dict, adapter_path: str, use_merged: bool, merged_path: str):
    if use_merged:
        model, tokenizer = _load_base(merged_path, cfg, quantize=False, device_map="auto")
    else:
        from peft import PeftModel
        model, tokenizer = _load_base(cfg["model"]["name"], cfg, quantize=True, device_map="auto")
        model = PeftModel.from_pretrained(model, adapter_path)

        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Per-example generation
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_SYSTEM_PROMPT = (
    "You are a compassionate and knowledgeable support assistant for caregivers "
    "of Alzheimer's and dementia patients. Your role is to provide empathetic, "
    "practical, and supportive advice. Acknowledge the caregiver's feelings first, "
    "then offer concrete suggestions. Keep responses warm, clear, and respectful."
)


def build_chat_prompt(tokenizer, user_message: str) -> str:
    """Use Qwen2.5's native ChatML template for inference.

    The Alpaca template (### Instruction / ### Response) was used during
    training but the base model's deep instruction-following ability is
    better activated by its original <|im_start|>/<|im_end|> format.
    """
    messages = [
        {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_message.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=900
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(
    model,
    tokenizer,
    records: list[dict],
    max_seq_length: int = 512,   # reduced from 1024 to limit logit tensor size
    batch_size: int = 1,         # reduced to 1 to avoid OOM on small GPUs
) -> float:
    """
    Compute average per-token perplexity on gold (input→output) pairs.
    Perplexity = exp(average negative log-likelihood per token).
    Lower is better.

    Note: Qwen2.5 has a 151,936-token vocabulary.  Casting logits to
    float32 requires  batch_size × seq_len × 151936 × 4 bytes.  On a
    low-VRAM GPU keep batch_size=1 and max_seq_length<=512.
    """
    import math
    import gc

    # Free any cached allocations left over from generation / BERTScore
    gc.collect()
    torch.cuda.empty_cache()

    total_nll = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(records), batch_size), desc="  Perplexity"):
        batch = records[i : i + batch_size]
        texts = [
            format_prompt_for_training(r, eos_token=tokenizer.eos_token)
            for r in batch
        ]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**enc, labels=enc["input_ids"])
            # outputs.loss is mean NLL over non-padding tokens
            nll = outputs.loss.item()
            n_tokens = enc["attention_mask"].sum().item()
            total_nll += nll * n_tokens
            total_tokens += n_tokens

        # Release the batch tensors immediately to keep peak VRAM low
        del enc, outputs
        torch.cuda.empty_cache()

    return math.exp(total_nll / total_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k].append(scores[k].fmeasure)

    return {k: round(sum(v) / len(v), 4) for k, v in agg.items()}


# ─────────────────────────────────────────────────────────────────────────────
# BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 16,
) -> dict:
    from bert_score import score as bert_score

    # Run BERTScore on CPU to avoid displacing the main model from VRAM
    P, R, F1 = bert_score(
        predictions,
        references,
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
        device="cpu",
    )
    return {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    eval_cfg = cfg["evaluation"]
    inf_cfg = cfg["inference"]

    test_file = args.test_file or eval_cfg["test_file"]
    predictions_file = eval_cfg["predictions_file"]
    results_file = eval_cfg["results_file"]
    adapter_path = args.adapter_path or inf_cfg["adapter_path"]
    merged_path = inf_cfg["merged_path"]
    use_merged = args.use_merged or inf_cfg.get("use_merged", False)
    max_eval_samples = args.max_eval_samples

    print("\n" + "=" * 65)
    print("  Evaluating Fine-Tuned Caregiver Chatbot")
    print("=" * 65)

    # ── Load test set ─────────────────────────────────────────────────────────
    if not Path(test_file).exists():
        sys.exit(f"ERROR: Test file not found: {test_file}")

    test_records = load_jsonl(test_file)
    if max_eval_samples and len(test_records) > max_eval_samples:
        import random
        random.seed(cfg["data"]["seed"])
        test_records = random.sample(test_records, max_eval_samples)
    print(f"  Evaluating on {len(test_records):,} test examples")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model(cfg, adapter_path, use_merged, merged_path)

    # ── Generate predictions ──────────────────────────────────────────────────
    predictions: list[str] = []
    references: list[str] = []
    pred_records: list[dict] = []

    if args.skip_generation and Path(predictions_file).exists():
        print(f"\nLoading cached predictions from {predictions_file} ...")
        for pr in load_jsonl(predictions_file):
            predictions.append(pr["prediction"])
            references.append(pr["reference"])
            pred_records.append(pr)
        print(f"  Loaded {len(predictions):,} cached predictions")
    else:
        print("\nGenerating predictions...")
        for rec in tqdm(test_records, desc="  Generating"):
            # Use Qwen's native chat template — consistent with inference
            prompt = build_chat_prompt(tokenizer, rec["input"])
            pred = generate_response(model, tokenizer, prompt, max_new_tokens=512)
            predictions.append(pred)
            references.append(rec["output"])
            pred_records.append(
                {
                    "instruction": rec["instruction"],
                    "input": rec["input"],
                    "reference": rec["output"],
                    "prediction": pred,
                }
            )

        # Save predictions
        Path(predictions_file).parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_file, "w", encoding="utf-8") as f:
            for pr in pred_records:
                f.write(json.dumps(pr, ensure_ascii=False) + "\n")
        print(f"  Predictions saved → {predictions_file}")

    # ── ROUGE ─────────────────────────────────────────────────────────────────
    print("\nComputing ROUGE scores...")
    rouge = compute_rouge(predictions, references)
    print(
        f"  ROUGE-1: {rouge['rouge1']:.4f}  |  "
        f"ROUGE-2: {rouge['rouge2']:.4f}  |  "
        f"ROUGE-L: {rouge['rougeL']:.4f}"
    )

    # ── BERTScore ─────────────────────────────────────────────────────────────
    print("\nComputing BERTScore (this downloads DistilBERT on first run)...")
    bs = compute_bertscore(
        predictions,
        references,
        model_type=eval_cfg.get("bertscore_model", "distilbert-base-uncased"),
        batch_size=eval_cfg.get("batch_size", 8),
    )
    print(
        f"  BERTScore P: {bs['precision']:.4f}  |  "
        f"R: {bs['recall']:.4f}  |  "
        f"F1: {bs['f1']:.4f}"
    )

    # ── Perplexity ────────────────────────────────────────────────────────────
    # Use a reduced seq_length for perplexity to keep logit tensors small.
    # Qwen2.5 vocab=151936: at batch=1, seq=512 the float32 logit cast
    # needs ~512*151936*4 ≈ 300 MB, which fits on a 4 GB GPU.
    ppl_max_seq = min(512, cfg["training"]["max_seq_length"])
    print(f"\nComputing Perplexity on test set (max_seq={ppl_max_seq}, batch=1)...")
    ppl = compute_perplexity(
        model,
        tokenizer,
        test_records,
        max_seq_length=ppl_max_seq,
        batch_size=1,
    )
    print(f"  Perplexity: {ppl:.2f}  (lower is better)")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "n_examples": len(test_records),
        "rouge": rouge,
        "bertscore": bs,
        "perplexity": round(ppl, 4),
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {results_file}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("EVALUATION SUMMARY")
    print("─" * 65)
    print(f"  ROUGE-1     : {rouge['rouge1']:.4f}")
    print(f"  ROUGE-2     : {rouge['rouge2']:.4f}")
    print(f"  ROUGE-L     : {rouge['rougeL']:.4f}")
    print(f"  BERTScore F1: {bs['f1']:.4f}")
    print(f"  Perplexity  : {ppl:.2f}")
    print("─" * 65)

    # Show a random example
    import random
    ex = random.choice(pred_records)
    print("\n--- Sample prediction ---")
    print(f"  Input     : {ex['input'][:100].strip()}...")
    print(f"  Reference : {ex['reference'][:150].strip()}")
    print(f"  Prediction: {ex['prediction'][:150].strip()}")
    print("─" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned chatbot.")
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--use_merged", action="store_true")
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Cap the number of test examples (for quick evaluation)",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation and reuse cached predictions.jsonl (saves ~3-4 hours)",
    )
    args = parser.parse_args()
    main(args)
