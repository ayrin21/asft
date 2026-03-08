"""
evaluate.py — Evaluate a fine-tuned ASFT model on a test JSON split.

Metrics:
  - Text Similarity (TS): token-level Jaccard similarity
  - Word F1 (W-F1)
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Exact Match (EM)

Usage:
    python evaluate.py \
        --model_dir ./outputs \
        --test_data ./data/test.json \
        --output_json ./results/eval_results.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def text_similarity(pred: str, gt: str) -> float:
    p, g = set(pred.lower().split()), set(gt.lower().split())
    if not g:
        return 1.0 if not p else 0.0
    return len(p & g) / len(p | g) if (p | g) else 1.0


def word_f1(pred: str, gt: str) -> float:
    p_tokens = pred.lower().split()
    g_tokens = gt.lower().split()
    if not g_tokens:
        return 1.0 if not p_tokens else 0.0
    common = sum(min(p_tokens.count(w), g_tokens.count(w)) for w in set(g_tokens))
    precision = common / len(p_tokens) if p_tokens else 0.0
    recall    = common / len(g_tokens) if g_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cer(pred: str, gt: str) -> float:
    """Levenshtein-based Character Error Rate."""
    if not gt:
        return 0.0 if not pred else 1.0
    p, g = list(pred), list(gt)
    dp = list(range(len(g) + 1))
    for pc in p:
        new_dp = [dp[0] + 1]
        for j, gc in enumerate(g):
            new_dp.append(min(new_dp[-1] + 1, dp[j + 1] + 1, dp[j] + (pc != gc)))
        dp = new_dp
    return dp[-1] / len(g)


def wer(pred: str, gt: str) -> float:
    """Levenshtein-based Word Error Rate."""
    p, g = pred.split(), gt.split()
    if not g:
        return 0.0 if not p else 1.0
    dp = list(range(len(g) + 1))
    for pw in p:
        new_dp = [dp[0] + 1]
        for j, gw in enumerate(g):
            new_dp.append(min(new_dp[-1] + 1, dp[j + 1] + 1, dp[j] + (pw != gw)))
        dp = new_dp
    return dp[-1] / len(g)


def score_tier(ts: float) -> str:
    if ts >= 0.90:
        return "Excellent"
    elif ts >= 0.75:
        return "Good"
    elif ts >= 0.50:
        return "Fair"
    return "Poor"


# ─────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert OCR system specialised in reading text from "
    "book covers that may be degraded by atmospheric turbulence blur. "
    "Extract ALL visible text exactly as it appears."
)


def predict(model, processor, image_path: str, max_new_tokens: int = 256) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [
             {"type": "image", "image": image},
             {"type": "text", "text": "Extract all text from this book cover."},
         ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    elapsed = time.time() - t0

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    pred = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return pred, elapsed


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",   type=str, required=True)
    parser.add_argument("--test_data",   type=str, required=True)
    parser.add_argument("--output_json", type=str, default="eval_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Load test split
    with open(args.test_data) as f:
        records = json.load(f)

    print(f"Evaluating {len(records)} samples...")
    results, all_metrics = [], {k: [] for k in ["ts", "wf1", "cer", "wer", "em", "time"]}

    for rec in tqdm(records):
        pred, elapsed = predict(model, processor, rec["image"], args.max_new_tokens)
        gt = rec["text"]

        m = {
            "ts":   text_similarity(pred, gt),
            "wf1":  word_f1(pred, gt),
            "cer":  cer(pred, gt),
            "wer":  wer(pred, gt),
            "em":   int(pred.strip().lower() == gt.strip().lower()),
            "time": elapsed,
        }
        for k, v in m.items():
            all_metrics[k].append(v)

        results.append({
            "image": rec["image"],
            "ground_truth": gt,
            "prediction": pred,
            "tier": score_tier(m["ts"]),
            **m,
        })

    # Aggregate
    n = len(records)
    summary = {
        "n_samples":    n,
        "text_sim":     sum(all_metrics["ts"])  / n * 100,
        "word_f1":      sum(all_metrics["wf1"]) / n * 100,
        "cer":          sum(all_metrics["cer"])  / n * 100,
        "wer":          sum(all_metrics["wer"])  / n * 100,
        "exact_match":  sum(all_metrics["em"])   / n * 100,
        "avg_inf_sec":  sum(all_metrics["time"]) / n,
        "tiers": {
            t: sum(1 for r in results if r["tier"] == t)
            for t in ["Excellent", "Good", "Fair", "Poor"]
        },
    }

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in summary.items():
        if k != "tiers":
            print(f"  {k:<20} {v:.2f}")
    print(f"\n  Tiers: {summary['tiers']}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"summary": summary, "per_sample": results}, f, indent=2)
    print(f"\n✅ Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
