"""
Autonomous Self-supervised Fine-Tuning (ASFT) for Qwen2.5-VL-3B
================================================================
Fine-tunes Qwen2.5-VL-3B-Instruct on turbulence-blurred book cover OCR
using QLoRA (4-bit NF4) with four novel components:
  - Autonomous Quality Evaluator (AQE)
  - Self-Critique Module (SCM)
  - Adaptive Curriculum Manager (ACM)
  - Composite loss: L = L_CE + 0.35*L_qual + 0.25*L_cons

Hardware: 2x Tesla T4 (Kaggle / Colab)
"""

import os
import json
import math
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from PIL import Image


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class ASFTConfig:
    # Model
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # QLoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    output_dir: str = "./outputs"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 200
    dataloader_num_workers: int = 2

    # Data
    train_data: str = "data/train.json"
    val_data: str = "data/val.json"

    # ASFT loss weights
    lambda_qual: float = 0.35
    lambda_cons: float = 0.25

    # ACM rolling window
    acm_window: int = 50


# ─────────────────────────────────────────────
# Autonomous Quality Evaluator (AQE)
# ─────────────────────────────────────────────

class AutonomousQualityEvaluator(nn.Module):
    """
    4 MLP heads scoring hidden-state representations across:
      sharpness, completeness, coherence, confidence.
    Outputs a 3-way decision: accept (0) / refine (1) / reject (2).
    Loss scaling: accept=1.0, refine=0.9, reject=1.5.
    """
    LOSS_SCALE = {0: 1.0, 1: 0.9, 2: 1.5}

    def __init__(self, hidden_size: int = 2048):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
            for _ in range(4)  # sharpness, completeness, coherence, confidence
        ])
        self.decision = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 3),  # accept / refine / reject
        )

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: (B, seq_len, hidden_size) → pool over seq
        pooled = hidden_states.mean(dim=1)  # (B, hidden_size)
        scores = torch.cat([h(pooled) for h in self.heads], dim=-1)  # (B, 4)
        logits = self.decision(scores)                                 # (B, 3)
        decisions = logits.argmax(dim=-1)                             # (B,)
        scales = torch.tensor(
            [self.LOSS_SCALE[d.item()] for d in decisions],
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        quality_loss = (1.0 - scores.mean(dim=-1)).mean()
        return decisions, scales, quality_loss


# ─────────────────────────────────────────────
# Self-Critique Module (SCM)
# ─────────────────────────────────────────────

class SelfCritiqueModule(nn.Module):
    """
    Lightweight 1-layer TransformerEncoder applied to hidden states
    at training time only — zero inference overhead.
    """
    def __init__(self, hidden_size: int = 2048, nhead: int = 8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        refined = self.encoder(hidden_states)
        return self.proj(refined)


# ─────────────────────────────────────────────
# Adaptive Curriculum Manager (ACM)
# ─────────────────────────────────────────────

class AdaptiveCurriculumManager:
    """
    Tracks per-sample difficulty using a rolling window of size `window`.
    Returns a difficulty score in [0, 1] for each sample.
    """
    def __init__(self, window: int = 50):
        self.window = window
        self.history: list[float] = []

    def update(self, loss_val: float):
        self.history.append(loss_val)
        if len(self.history) > self.window:
            self.history.pop(0)

    @property
    def difficulty(self) -> float:
        if not self.history:
            return 0.5
        mean = sum(self.history) / len(self.history)
        # Normalise to [0,1] with a soft sigmoid
        return float(torch.sigmoid(torch.tensor(mean - 1.0)).item())


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class BookCoverDataset(Dataset):
    """
    Expects a JSON file with records:
      {"image": "path/to/image.jpg", "text": "ground truth OCR text"}
    """
    SYSTEM_PROMPT = (
        "You are an expert OCR system specialised in reading text from "
        "book covers that may be degraded by atmospheric turbulence blur. "
        "Extract ALL visible text exactly as it appears."
    )

    def __init__(self, json_path: str, processor, max_length: int = 2048):
        with open(json_path) as f:
            self.records = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image"]).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text from this book cover."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rec["text"]}],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


# ─────────────────────────────────────────────
# ASFT Trainer
# ─────────────────────────────────────────────

class ASFTTrainer(Trainer):
    """
    Custom Trainer applying the composite ASFT loss:
      L = L_CE + λ_qual * L_qual + λ_cons * L_cons
    with AQE decision-conditioned loss scaling and SCM hidden-state refinement.
    """
    def __init__(self, *args, asft_config: ASFTConfig, aQE, scm, acm, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = asft_config
        self.aQE = aQE
        self.scm = scm
        self.acm = acm

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)

        # Standard cross-entropy loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # AQE: quality loss + loss scaling
        last_hidden = outputs.hidden_states[-1]          # (B, seq, H)
        decisions, scales, qual_loss = self.aQE(last_hidden)
        scaled_ce = (ce_loss * scales.mean())

        # SCM: consistency loss between original and refined hidden states
        refined = self.scm(last_hidden)
        cons_loss = nn.MSELoss()(refined, last_hidden.detach())

        # Composite loss
        loss = (
            scaled_ce
            + self.cfg.lambda_qual * qual_loss
            + self.cfg.lambda_cons * cons_loss
        )

        # ACM update
        self.acm.update(ce_loss.item())

        if return_outputs:
            return loss, outputs
        return loss


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    cfg = ASFTConfig()

    # ── Quantisation config ──────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Model & processor ───────────────────
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA ────────────────────────────────
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── ASFT modules ────────────────────────
    hidden_size = model.config.hidden_size  # 2048 for 3B
    aQE = AutonomousQualityEvaluator(hidden_size).to(model.device)
    scm = SelfCritiqueModule(hidden_size).to(model.device)
    acm = AdaptiveCurriculumManager(window=cfg.acm_window)

    # ── Datasets ────────────────────────────
    train_dataset = BookCoverDataset(cfg.train_data, processor, cfg.max_seq_length)
    val_dataset   = BookCoverDataset(cfg.val_data,   processor, cfg.max_seq_length)

    # ── Training args ───────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.save_steps,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    # ── Trainer ─────────────────────────────
    trainer = ASFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        asft_config=cfg,
        aQE=aQE,
        scm=scm,
        acm=acm,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print(f"✅ Model saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
