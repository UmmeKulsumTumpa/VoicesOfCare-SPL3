"""
utils/prompt_utils.py
=====================
Shared prompt-formatting utilities used across all training and
inference scripts.

Design notes:
  - We use the Alpaca instruction template because it is:
      a) model-agnostic (works with Phi-3, Phi-2, TinyLlama, Qwen2)
      b) well-understood by instruction-tuned base models
      c) easy for users to interact with at inference time
  - The EOS token is appended only during training so the model
    learns when to stop generating.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca-style template
# ─────────────────────────────────────────────────────────────────────────────
ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

# Full template including the output (used only during training)
ALPACA_TRAINING_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}{eos_token}"
)


def format_prompt_for_training(example: dict, eos_token: str = "</s>") -> str:
    """
    Format a single dataset example into the full Alpaca training string.

    Parameters
    ----------
    example : dict
        Must have keys: "instruction", "input", "output"
    eos_token : str
        The tokenizer's EOS token.  Appended after the response so the
        model learns to terminate generation.

    Returns
    -------
    str
        The fully formatted training string.
    """
    return ALPACA_TRAINING_TEMPLATE.format(
        instruction=example["instruction"].strip(),
        input=example["input"].strip(),
        output=example["output"].strip(),
        eos_token=eos_token,
    )


def format_prompt_for_inference(
    instruction: str,
    input_text: str,
) -> str:
    """
    Format a prompt for inference (no output / EOS token appended).

    Parameters
    ----------
    instruction : str
        The system-level instruction.
    input_text : str
        The structured patient context + Reddit post.

    Returns
    -------
    str
        The prompt string to feed the model.
    """
    return ALPACA_PROMPT_TEMPLATE.format(
        instruction=instruction.strip(),
        input=input_text.strip(),
    )


def build_inference_prompt(
    patient_age: str = "unknown",
    gender: str = "unknown",
    known_symptoms: list[str] | None = None,
    rare_symptoms: list[str] | None = None,
    caregiver_motivation: list[str] | None = None,
    caregiver_relationship: str = "unspecified",
    post_text: str = "",
) -> str:
    """
    Build a structured input string from individual fields, then wrap it
    in the inference prompt template.

    This mirrors exactly how the training data was structured, ensuring
    the model sees the same format it was trained on.

    Parameters
    ----------
    patient_age : str
    gender : str
    known_symptoms : list of str
    rare_symptoms : list of str
    caregiver_motivation : list of str
    caregiver_relationship : str
    post_text : str
        The caregiver's Reddit post content.

    Returns
    -------
    str
        The full prompt ready for tokenization and generation.
    """
    known_symptoms = known_symptoms or []
    rare_symptoms = rare_symptoms or []
    caregiver_motivation = caregiver_motivation or []

    structured_input = (
        f"Patient Age: {patient_age}\n"
        f"Gender: {gender}\n\n"
        f"Known Symptoms: {known_symptoms}\n"
        f"Rare Symptoms: {rare_symptoms}\n\n"
        f"Caregiver Motivation: {caregiver_motivation}\n"
        f"Caregiver Relationship: {caregiver_relationship}\n\n"
        f"Reddit Post:\n{post_text.strip()}"
    )

    instruction = (
        "Provide supportive and practical advice for a caregiver of a "
        "dementia or Alzheimer's patient."
    )

    return format_prompt_for_inference(instruction, structured_input)


def extract_response(generated_text: str) -> str:
    """
    Strip the prompt prefix from a generated string, returning only the
    model's response.

    Parameters
    ----------
    generated_text : str
        The raw output from model.generate() decoded to string.

    Returns
    -------
    str
        Only the text after "### Response:\n".
    """
    marker = "### Response:\n"
    if marker in generated_text:
        return generated_text.split(marker, 1)[1].strip()
    return generated_text.strip()
