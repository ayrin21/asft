# ASFT: Autonomous Self-supervised Fine-Tuning for Turbulence-Degraded OCR


Official implementation of **Autonomous Self-supervised Fine-Tuning (ASFT)** applied to `Qwen2.5-VL-3B-Instruct` for OCR on atmospheric turbulence-blurred book covers. In **qwen3b2-5-autonomous-qlora.ipynb** File you can see the Run file perfectly. Check it out let us know How we can more improve.

## Dataset Link in Kaggle

Kaggle-https://www.kaggle.com/datasets/fatehajannatayrin/turbulence-blur-text-extraction-book-images.
Demo link-  https://huggingface.co/spaces/Rafi123/qwen3b-autonomous-demo.

---

## Results

| Model | Text Sim ↑ | Word F1 ↑ | CER ↓ | WER ↓ | EM ↑ |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B (zero-shot) | 45.09% | 28.74% | 72.80% | 89.30% | 0% |
| Standard SFT | 76.40% | 72.30% | 31.20% | 38.60% | 18% |
| Standard QLoRA | 84.10% | 79.80% | 20.50% | 28.40% | 26% |
| **ASFT (ours)** | **93.27%** | **88.18%** | **9.59%** | **14.57%** | **38%** |

74% of test samples reach **Excellent** tier (≥ 90% text similarity).

---

## Method

ASFT adds four components on top of standard QLoRA (4-bit NF4, r=32):

| Component | Role | Overhead |
|---|---|---|
| **AQE** — Autonomous Quality Evaluator | 4 MLP heads; 3-way accept/refine/reject decision + loss scaling | Training only |
| **SCM** — Self-Critique Module | 1-layer TransformerEncoder refining hidden states | Training only (zero inference cost) |
| **ACM** — Adaptive Curriculum Manager | Rolling window difficulty scoring | Training only |
| **Composite Loss** | L = L_CE + 0.35·L_qual + 0.25·L_cons | — |

---

## Setup

### 1. Install dependencies

```bash
# Recommended: create a virtual environment first
python setup.py
```

Or manually:

```bash
pip install git+https://github.com/huggingface/transformers \
    accelerate peft bitsandbytes qwen-vl-utils Pillow tqdm
```

### 2. Prepare data

```bash
python prepare_data.py \
    --images_dir ./raw_images \
    --labels_csv ./labels.csv \
    --output_dir ./data \
    --alpha 0.7
```

Expected CSV format:
```
filename,text
cover_001.jpg,"The Great Gatsby F. Scott Fitzgerald"
cover_002.jpg,"1984 George Orwell"
```

### 3. Train

```bash
python train.py
```

Edit `ASFTConfig` at the top of `train.py` to adjust paths, rank, batch size, etc.

**Hardware:** Tested on 2× Tesla T4 (Kaggle / Colab). ~330 min for 2 epochs on 3,956 samples.

### 4. Evaluate

```bash
python evaluate.py \
    --model_dir ./outputs \
    --test_data ./data/test.json \
    --output_json ./results/eval_results.json
```

### 5. Inference

```bash
python inference.py \
    --model_dir ./outputs \
    --image path/to/blurred_cover.jpg
```

---

## Repository Structure

```
asft/
├── train.py          # Full ASFT training pipeline (AQE + SCM + ACM + composite loss)
├── inference.py      # Single-image inference
├── evaluate.py       # Batch evaluation with all metrics
├── prepare_data.py   # Turbulence simulation + train/val/test splits
├── setup.py          # Dependency installer
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Configuration

Key parameters in `ASFTConfig` (top of `train.py`):

| Parameter | Default | Description |
|---|---|---|
| `model_id` | `Qwen/Qwen2.5-VL-3B-Instruct` | Base model |
| `lora_r` | `32` | LoRA rank |
| `lora_alpha` | `64` | LoRA scaling |
| `lambda_qual` | `0.35` | AQE loss weight |
| `lambda_cons` | `0.25` | SCM consistency loss weight |
| `acm_window` | `50` | ACM rolling window size |
| `max_seq_length` | `2048` | Maximum token length |

---
