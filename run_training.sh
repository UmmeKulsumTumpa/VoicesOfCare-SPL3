#!/usr/bin/env bash
# =============================================================
# run_training.sh
# Full end-to-end training pipeline for the Alzheimer/Dementia
# Caregiver Chatbot fine-tuning.
#
# Usage:
#   chmod +x run_training.sh
#   ./run_training.sh              # full pipeline
#   ./run_training.sh --smoke      # quick smoke-test (50 steps)
# =============================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[1;34m'; NC='\033[0m'

log()   { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# ── Smoke-test flag ───────────────────────────────────────────
SMOKE=false
MAX_STEPS=""
MAX_SAMPLES=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE=true
    MAX_STEPS="--max_steps 50"
    MAX_SAMPLES="--max_samples 500"
    warn "SMOKE TEST MODE: 500 samples, 50 training steps only"
fi

# ── Verify we're in the training/ directory ───────────────────
if [[ ! -f "config/training_config.yaml" ]]; then
    error "Please run this script from the training/ directory.\n  cd training && ./run_training.sh"
fi

# ── Check GPU ─────────────────────────────────────────────────
log "Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {name}  |  VRAM: {mem:.1f} GB')
    if mem < 3.5:
        print('  WARNING: Less than 3.5 GB VRAM detected.')
        print('  Consider switching to TinyLlama-1.1B in config.')
else:
    print('  WARNING: No CUDA GPU detected. Training will be very slow on CPU.')
"

# ── Step 0: Install dependencies ─────────────────────────────
log "Step 0 — Installing Python dependencies..."
if ! python3 -c "import torch" &>/dev/null; then
    warn "torch not found. Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 --quiet
fi

pip install -r requirements.txt --quiet
ok "Dependencies installed"

# ── Step 1: Prepare dataset ───────────────────────────────────
log "Step 1 — Preparing dataset..."
python3 scripts/01_prepare_dataset.py \
    --config config/training_config.yaml \
    ${MAX_SAMPLES}
ok "Dataset prepared"

# ── Step 2: QLoRA Training ────────────────────────────────────
log "Step 2 — Starting QLoRA fine-tuning..."
log "  This will take ~2-4 hours for 3 epochs on RTX 3050 (full dataset)"
log "  Monitor GPU usage: watch -n 2 nvidia-smi"

python3 scripts/02_train_qlora.py \
    --config config/training_config.yaml \
    ${MAX_STEPS}
ok "Training complete"

# ── Step 3: Merge adapter ────────────────────────────────────
if [[ "$SMOKE" == false ]]; then
    log "Step 3 — Merging LoRA adapter into base model..."
    python3 scripts/03_merge_adapter.py \
        --config config/training_config.yaml
    ok "Adapter merged"
else
    warn "Smoke test: skipping merge step"
fi

# ── Step 4: Evaluate ─────────────────────────────────────────
log "Step 4 — Running evaluation on test set..."
EVAL_EXTRA=""
if [[ "$SMOKE" == true ]]; then
    EVAL_EXTRA="--max_eval_samples 50"
fi

python3 scripts/05_evaluate.py \
    --config config/training_config.yaml \
    ${EVAL_EXTRA}
ok "Evaluation complete"

# ── Done ─────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Pipeline complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo "  To chat with the model:"
echo "    python3 scripts/04_chat_inference.py --mode freetext"
echo "    python3 scripts/04_chat_inference.py --mode structured"
echo ""
echo "  Results saved in:"
echo "    output/run_01/        ← LoRA adapter"
echo "    output/merged/        ← merged model"
echo "    output/eval_results.json ← metrics"
echo ""
