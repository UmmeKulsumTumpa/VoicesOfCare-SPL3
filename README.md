# SPL-Bot — Fine-Tuning Pipeline

QLoRA fine-tuning of a small language model on Reddit caregiver conversations (r/Alzheimers, r/dementia) to build an empathetic support chatbot.

---

## Structure

```
training/
├── config/training_config.yaml   # all hyperparameters — edit this first
├── scripts/
│   ├── 01_prepare_dataset.py     # load, deduplicate, split JSONL data
│   ├── 02_train_qlora.py         # QLoRA fine-tuning
│   ├── 03_merge_adapter.py       # fuse LoRA weights into base model
│   ├── 04_chat_inference.py      # interactive terminal chatbot
│   ├── 05_evaluate.py            # ROUGE + BERTScore + Perplexity
│   └── 06_hf_inference.py        # inference via HuggingFace Hub
├── utils/
│   ├── model_loader.py
│   └── prompt_utils.py           # Alpaca-style prompt formatting
├── processed_data/               # generated — train/val/test splits
├── output/                       # generated — checkpoints & merged model
├── requirements.txt
└── run_training.sh               # end-to-end pipeline (one command)
```

---

## Model

**Base:** `Qwen/Qwen2.5-3B-Instruct` — 3.1B params, ~2.0 GB VRAM in 4-bit.  
**Target hardware:** NVIDIA RTX 3050 Mobile (4 GB VRAM).

QLoRA settings: 4-bit NF4, double quantization, LoRA r=16 α=32, paged AdamW 8-bit.  
Trainable params: ~21M / 3.1B (~0.7%).

---

## Setup

```bash
# 1. Create and activate virtualenv
python3 -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch with CUDA (adjust cu121 to match your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. HuggingFace login (needed for gated models)
huggingface-cli login
```

---

## Usage

```bash
# Full pipeline (recommended)
chmod +x run_training.sh
./run_training.sh

# Quick smoke-test (500 samples, 50 steps)
./run_training.sh --smoke

# Or run steps individually
python scripts/01_prepare_dataset.py
python scripts/02_train_qlora.py
python scripts/03_merge_adapter.py
python scripts/04_chat_inference.py   # interactive chat
python scripts/05_evaluate.py
```

---

## Data

Raw JSONL files in `../data/one-bucket-top3-comments/`.  
Splits: 90% train / 5% val / 5% test (seed 42).  
Prompt format: Alpaca instruction template; loss computed on `### Response:` tokens only.

---

## Output

| Path | Contents |
|---|---|
| `output/run_XX/` | LoRA adapter weights + tokenizer |
| `output/merged/` | Full merged model (base + adapter) |
| `output/eval_results.json` | ROUGE / BERTScore / Perplexity |
| `output/predictions.jsonl` | Model predictions on test set |
