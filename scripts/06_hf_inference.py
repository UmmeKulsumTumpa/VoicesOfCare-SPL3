#!/usr/bin/env python3
"""
scripts/06_hf_inference.py
===========================
Interactive chatbot using the Hugging Face Inference API (pipeline endpoint).
No model download, no GPU required — calls HF's hosted endpoint directly.

Works with any model on the Hub (public or private with a token).

Requirements:  pip install requests  (already in venv via transformers)

Get a free token at: https://huggingface.co/settings/tokens
Set it once:
  huggingface-cli login
  OR export HF_TOKEN=hf_...

Usage:
  .venv/bin/python scripts/06_hf_inference.py
  .venv/bin/python scripts/06_hf_inference.py --mode structured
  .venv/bin/python scripts/06_hf_inference.py --repo IAmPilgrim/caregiver-chatbot-qwen25-3b
"""

from __future__ import annotations

import argparse
import os
import time

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_REPO = "IAmPilgrim/caregiver-chatbot-qwen25-3b"
HF_API_URL   = "https://api-inference.huggingface.co/models/{repo}"

SYSTEM_PROMPT = (
    "You are a compassionate and knowledgeable support assistant for caregivers "
    "of Alzheimer's and dementia patients. Your role is to provide empathetic, "
    "practical, and supportive advice. Acknowledge the caregiver's feelings first, "
    "then offer concrete suggestions. Keep responses warm, clear, and respectful."
)

GEN_PARAMS = dict(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.3,
    return_full_text=False,   # return only the generated part, not the prompt
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder  (Qwen2.5 ChatML format)
# ─────────────────────────────────────────────────────────────────────────────

def build_chatml_prompt(user_message: str) -> str:
    """Format using Qwen2.5's native <|im_start|>...<|im_end|> template."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def generate(repo: str, user_message: str, token: str | None) -> str:
    url     = HF_API_URL.format(repo=repo)
    prompt  = build_chatml_prompt(user_message)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "inputs": prompt,
        "parameters": GEN_PARAMS,
        "options": {"wait_for_model": True},   # wait if model is loading (cold start)
    }

    for attempt in range(3):
        resp = requests.post(url, headers=headers, json=payload, timeout=120)

        if resp.status_code == 503:
            # Model is loading on HF side — wait and retry
            wait = resp.json().get("estimated_time", 20)
            print(f"  [Model loading on HF, waiting {wait:.0f}s...]")
            time.sleep(min(wait, 30))
            continue

        if resp.status_code == 401:
            raise RuntimeError(
                "HuggingFace returned 401 Unauthorized.\n"
                "Run: huggingface-cli login   OR   export HF_TOKEN=hf_..."
            )

        if resp.status_code == 410:
            raise RuntimeError(
                "HTTP 410 Gone: HuggingFace free serverless Inference API no longer supports\n"
                "loading arbitrary fine-tuned models on demand.\n\n"
                "Free alternatives:\n"
                "  1. Google Colab (free T4 GPU):\n"
                "       https://colab.research.google.com — paste the snippet from the README\n"
                "  2. Run locally with the adapter (4-bit, fits your RTX 3050):\n"
                "       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n"
                "       .venv/bin/python scripts/04_chat_inference.py \\\n"
                "           --config config/training_config.yaml \\\n"
                "           --adapter_path ./output/run_02 --mode freetext\n"
                "  3. Paid HF Inference Endpoint (~$0.06/hr):\n"
                "       https://huggingface.co/inference-endpoints\n"
            )

        resp.raise_for_status()

        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        raise RuntimeError(f"Unexpected API response: {data}")

    raise RuntimeError("Model did not become available after 3 retries.")

# ─────────────────────────────────────────────────────────────────────────────
# Interaction modes
# ─────────────────────────────────────────────────────────────────────────────

def freetext_chat(repo: str, token: str | None) -> None:
    print("═" * 60)
    print("  Alzheimer / Dementia Caregiver Support Chatbot")
    print("  Powered by HuggingFace Inference API")
    print("  (type 'quit' to exit)")
    print("═" * 60)

    while True:
        try:
            user_text = input("\nYou: ").strip()
            if user_text.lower() == "quit":
                raise KeyboardInterrupt
            if not user_text:
                continue

            print("[Generating response...]")
            response = generate(repo, user_text, token)
            print("\n" + "─" * 60)
            print(f"Chatbot:\n{response}")
            print("─" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye. Take care of yourself too. 💙")
            break


def structured_chat(repo: str, token: str | None) -> None:
    print("═" * 60)
    print("  Alzheimer / Dementia Caregiver Support Chatbot")
    print("  Powered by HuggingFace Inference API  |  Structured Mode")
    print("  (Ctrl+C to exit)")
    print("═" * 60)

    while True:
        try:
            print("\n── New conversation ──────────────────────────────────")
            age          = input("Patient age (or 'unknown'): ").strip() or "unknown"
            gender       = input("Patient gender (male/female/unknown): ").strip() or "unknown"
            sym_input    = input("Known symptoms (comma-separated, or leave blank): ").strip()
            symptoms     = [s.strip() for s in sym_input.split(",") if s.strip()]
            motivation   = input("Why are you posting? (e.g. Seeking_Emotional_Support): ").strip()
            relationship = input("Your relationship to the patient: ").strip() or "unspecified"
            post_text    = input("\nYour message: ").strip()

            if not post_text:
                print("[No message entered. Skipping.]")
                continue

            user_message = (
                f"Patient Age: {age}\nGender: {gender}\n\n"
                f"Known Symptoms: {symptoms}\n"
                f"Caregiver Motivation: {[motivation] if motivation else []}\n"
                f"Caregiver Relationship: {relationship}\n\n"
                f"{post_text}"
            )
            print("\n[Generating response...]")
            response = generate(repo, user_message, token)
            print("\n" + "─" * 60)
            print(f"Chatbot:\n{response}")
            print("─" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye. Take care of yourself too. 💙")
            break

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caregiver chatbot via HuggingFace Inference API (no local GPU needed)."
    )
    parser.add_argument("--repo",  type=str, default=DEFAULT_REPO,
                        help=f"HuggingFace repo ID (default: {DEFAULT_REPO})")
    parser.add_argument("--mode",  choices=["freetext", "structured"], default="freetext")
    parser.add_argument("--token", type=str, default=None,
                        help="HF token (fallback: HF_TOKEN env var or ~/.cache/huggingface/token)")
    args = parser.parse_args()

    # Resolve token: CLI arg > env var > cached login
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        token_file = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_file):
            with open(token_file) as f:
                token = f.read().strip() or None

    print(f"\n  Repo  : {args.repo}")
    print(f"  Mode  : {args.mode}")
    print(f"  Token : {'set ✓' if token else 'NOT SET — public models only'}\n")

    if args.mode == "structured":
        structured_chat(args.repo, token)
    else:
        freetext_chat(args.repo, token)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_REPO = "IAmPilgrim/caregiver-chatbot-qwen25-3b"

SYSTEM_PROMPT = (
    "You are a compassionate and knowledgeable support assistant for caregivers "
    "of Alzheimer's and dementia patients. Your role is to provide empathetic, "
    "practical, and supportive advice. Acknowledge the caregiver's feelings first, "
    "then offer concrete suggestions. Keep responses warm, clear, and respectful."
)

GEN_PARAMS = dict(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def generate(client: InferenceClient, repo: str, user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message.strip()},
    ]
    response = client.chat_completion(
        messages=messages,
        model=repo,
        **GEN_PARAMS,
    )
    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Interaction modes
# ─────────────────────────────────────────────────────────────────────────────

def freetext_chat(client: InferenceClient, repo: str) -> None:
    print("═" * 60)
    print("  Alzheimer / Dementia Caregiver Support Chatbot")
    print("  Powered by HuggingFace Inference API")
    print("  (type 'quit' to exit)")
    print("═" * 60)

    while True:
        try:
            user_text = input("\nYou: ").strip()
            if user_text.lower() == "quit":
                raise KeyboardInterrupt
            if not user_text:
                continue

            print("[Generating response...]")
            response = generate(client, repo, user_text)
            print("\n" + "─" * 60)
            print(f"Chatbot:\n{response}")
            print("─" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye. Take care of yourself too. 💙")
            break


def structured_chat(client: InferenceClient, repo: str) -> None:
    print("═" * 60)
    print("  Alzheimer / Dementia Caregiver Support Chatbot")
    print("  Powered by HuggingFace Inference API  |  Structured Mode")
    print("  (Ctrl+C to exit)")
    print("═" * 60)

    while True:
        try:
            print("\n── New conversation ──────────────────────────────────")
            age          = input("Patient age (or 'unknown'): ").strip() or "unknown"
            gender       = input("Patient gender (male/female/unknown): ").strip() or "unknown"
            sym_input    = input("Known symptoms (comma-separated, or leave blank): ").strip()
            symptoms     = [s.strip() for s in sym_input.split(",") if s.strip()]
            motivation   = input("Why are you posting? (e.g. Seeking_Emotional_Support): ").strip()
            relationship = input("Your relationship to the patient: ").strip() or "unspecified"
            post_text    = input("\nYour message: ").strip()

            if not post_text:
                print("[No message entered. Skipping.]")
                continue

            user_message = (
                f"Patient Age: {age}\nGender: {gender}\n\n"
                f"Known Symptoms: {symptoms}\n"
                f"Caregiver Motivation: {[motivation] if motivation else []}\n"
                f"Caregiver Relationship: {relationship}\n\n"
                f"{post_text}"
            )
            print("\n[Generating response...]")
            response = generate(client, repo, user_message)
            print("\n" + "─" * 60)
            print(f"Chatbot:\n{response}")
            print("─" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye. Take care of yourself too. 💙")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caregiver chatbot via HuggingFace Inference API (no local GPU needed)."
    )
    parser.add_argument(
        "--repo", type=str, default=DEFAULT_REPO,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO})"
    )
    parser.add_argument(
        "--mode", choices=["freetext", "structured"], default="freetext",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HF token (fallback: HF_TOKEN env var or ~/.cache/huggingface/token)"
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    client = InferenceClient(token=token)

    print(f"\n  Repo  : {args.repo}")
    print(f"  Mode  : {args.mode}")
    print(f"  Token : {'set' if token else 'from ~/.cache/huggingface/token (huggingface-cli login)'}\n")

    if args.mode == "structured":
        structured_chat(client, args.repo)
    else:
        freetext_chat(client, args.repo)


if __name__ == "__main__":
    main()
