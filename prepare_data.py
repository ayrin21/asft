"""
prepare_data.py — Generate turbulence-blurred book cover dataset.

Applies the P-Turbulence simulation (α=0.7) to a folder of clean book cover
images and produces train/val/test JSON splits compatible with BookCoverDataset.

Usage:
    python prepare_data.py \
        --images_dir ./raw_images \
        --labels_csv ./labels.csv \
        --output_dir ./data \
        --train_ratio 0.8 \
        --val_ratio 0.1
"""

import os
import json
import argparse
import random
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────
# Turbulence simulation (P-Turbulence, α=0.7)
# ─────────────────────────────────────────────

def generate_turbulence_psf(size: int = 64, alpha: float = 0.7) -> np.ndarray:
    """
    Generate a turbulence Point Spread Function using the Kolmogorov
    power spectrum model with strength parameter alpha.
    """
    freq = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq, freq)
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1e-10  # avoid division by zero

    # Kolmogorov power spectrum: P(f) ∝ f^(-11/3) * alpha
    power = np.exp(-alpha * (f ** (11 / 3)))
    phase = 2 * np.pi * np.random.rand(size, size)
    psf_freq = power * np.exp(1j * phase)
    psf = np.abs(np.fft.ifft2(psf_freq))
    psf /= psf.sum()
    return psf.astype(np.float32)


def apply_turbulence(image: Image.Image, alpha: float = 0.7, seed: int = None) -> Image.Image:
    """Apply turbulence blur to a PIL image."""
    if seed is not None:
        np.random.seed(seed)

    img_array = np.array(image).astype(np.float32) / 255.0
    psf = generate_turbulence_psf(size=64, alpha=alpha)
    blurred = np.zeros_like(img_array)

    for c in range(img_array.shape[2]):
        channel_fft = np.fft.fft2(img_array[:, :, c])
        psf_padded = np.zeros_like(img_array[:, :, c])
        h, w = psf.shape
        psf_padded[:h, :w] = psf
        psf_fft = np.fft.fft2(psf_padded)
        blurred[:, :, c] = np.abs(np.fft.ifft2(channel_fft * psf_fft))

    blurred = np.clip(blurred * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blurred)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",  type=str, required=True)
    parser.add_argument("--labels_csv",  type=str, required=True,
                        help="CSV with columns: filename, text")
    parser.add_argument("--output_dir",  type=str, default="./data")
    parser.add_argument("--blurred_dir", type=str, default="./data/blurred")
    parser.add_argument("--alpha",       type=float, default=0.7)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.blurred_dir, exist_ok=True)

    # Load labels
    labels = {}
    with open(args.labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["filename"]] = row["text"]

    # Process images
    records = []
    image_files = sorted(Path(args.images_dir).glob("*.jpg")) + \
                  sorted(Path(args.images_dir).glob("*.png"))

    print(f"Processing {len(image_files)} images with α={args.alpha}...")
    for img_path in tqdm(image_files):
        fname = img_path.name
        if fname not in labels:
            continue

        blurred_path = os.path.join(args.blurred_dir, fname)
        if not os.path.exists(blurred_path):
            img = Image.open(img_path).convert("RGB")
            blurred = apply_turbulence(img, alpha=args.alpha)
            blurred.save(blurred_path)

        records.append({"image": blurred_path, "text": labels[fname]})

    # Split
    random.shuffle(records)
    n = len(records)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)

    splits = {
        "train": records[:n_train],
        "val":   records[n_train:n_train + n_val],
        "test":  records[n_train + n_val:],
    }

    for split, data in splits.items():
        out_path = os.path.join(args.output_dir, f"{split}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {split}: {len(data)} samples → {out_path}")

    print("\n✅ Data preparation complete.")


if __name__ == "__main__":
    main()
