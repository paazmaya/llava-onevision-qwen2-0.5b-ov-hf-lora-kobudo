#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Kobudo/Karate Item Recognition
Using PEFT with LLaVA-OneVision models

Optimized for: Windows 11, NVIDIA GTX 4070 (12GB VRAM)
Base Model: llava-hf/llava-onevision-qwen2-0.5b-ov-hf
Data Format: Folder-based images
Language: English only

Usage:
    Full training run (default hyperparameters):
        uv run python train_kobudo_lora.py

    Smoke test — verifies the full pipeline in ~5 steps before committing to a long run:
        uv run python train_kobudo_lora.py --smoke-test

    Custom training:
        uv run python train_kobudo_lora.py --data-path ./my_images --epochs 5 --lr 5e-5

Smoke test mode
---------------
Pass --smoke-test to do a quick sanity check without wasting GPU hours. It caps
the dataset at 8 samples, limits training to 5 optimizer steps, reduces sequence
length and batch size, and skips saving the final model. Every stage of the
pipeline (data loading, tokenisation, forward pass, backward pass, evaluation)
is exercised so any configuration error surfaces immediately.

Item definitions:
    Edit items.toml to add, remove, or re-categorise kobudo/karate items
    without touching this script.

Requirements:
    uv sync
"""

import os
import json
import tomllib
import argparse
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

import torch
from PIL import Image
from torch.utils.data import Dataset

# Path to the TOML file that defines item keywords and their descriptions.
# Edit items.toml to add, remove, or re-categorise items without touching code.
_ITEMS_TOML = Path(__file__).parent / "items.toml"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for LoRA training - optimized for 12GB VRAM"""

    base_model: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    data_path: str = "./training_data"
    output_dir: str = "./output/kobudo_lora"
    max_new_tokens: int = 512
    max_seq_length: int = 8192
    temperature: float = 0.1
    top_p: float = 0.9
    mixed_precision: str = "bf16"
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    seed: int = 42
    num_workers: int = 2
    smoke_test: bool = False

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=1)
def _load_items() -> tuple[dict[str, str], str, str]:
    """
    Load item keywords and caption text from items.toml (read once, then cached).

    Returns:
        items    — merged dict of keyword -> description from all non-[general] sections
        fallback — caption used when no keyword matches the filename
        prefix   — short string prepended to every matched-item caption
    """
    with open(_ITEMS_TOML, "rb") as f:
        data = tomllib.load(f)

    general = data.get("general", {})
    fallback = general.get(
        "fallback_caption",
        "Describe this Okinawan martial arts equipment in detail, "
        "including its name, type, traditional use, and any distinguishing "
        "features. Respond in English only.",
    )
    prefix = general.get(
        "prompt_prefix", "Describe this Okinawan martial arts item in detail."
    )

    items: dict[str, str] = {}
    for section, entries in data.items():
        if section != "general" and isinstance(entries, dict):
            items.update(entries)

    return items, fallback, prefix


def auto_generate_caption(filename: str) -> str:
    """
    Generate English caption based on filename for kobudo/karate items.
    Keywords and descriptions are loaded from items.toml; all category
    sections are merged into one lookup so the file stays the single source
    of truth for item data.
    """
    filename_lower = filename.lower().replace("_", " ").replace("-", " ")

    items, fallback, prefix = _load_items()

    for item_name, description in items.items():
        if item_name in filename_lower:
            return f"{prefix} {description}"

    return fallback


class KobudoDataset(Dataset):
    """Dataset for kobudo/karate items from folder-based images"""

    def __init__(self, image_folder: str, processor: Any, max_length: int = 2048):
        self.image_folder = Path(image_folder)
        self.processor = processor
        self.max_length = max_length
        self.image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        # Find all images
        self.image_files = []
        for ext in self.image_extensions:
            self.image_files.extend(self.image_folder.glob(f"*{ext}"))
            self.image_files.extend(self.image_folder.glob(f"*{ext.upper()}"))

        self.image_files = sorted(self.image_files)

        if len(self.image_files) == 0:
            raise ValueError(
                f"No images found in {image_folder}. "
                f"Supported formats: JPG, JPEG, PNG, WebP, BMP"
            )

        logger.info(f"Found {len(self.image_files)} images in {image_folder}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_files[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            image = Image.new("RGB", (336, 336), color="black")

        # Generate English caption
        caption = auto_generate_caption(image_path.stem)

        inputs = self.processor(
            text=[caption],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}


def load_model_and_processor(config: TrainingConfig):
    """Load model and processor with LoRA applied"""
    from transformers import AutoProcessor, AutoModelForVision2Seq  # type: ignore[attr-defined]
    from peft import LoraConfig, get_peft_model

    logger.info(f"Loading base model: {config.base_model}")

    processor = AutoProcessor.from_pretrained(config.base_model, trust_remote_code=True)

    torch_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

    model = AutoModelForVision2Seq.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="VISION_LANG_LOSS",
    )

    model = get_peft_model(model, lora_config)

    trainable_params, total_params = 0, 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    logger.info(
        f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    return model, processor


def train(config: TrainingConfig):
    """Main training function"""
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForVision2Seq,
        set_seed,
    )  # type: ignore[attr-defined]
    from torch.utils.data import Subset
    import random

    logger.info("=" * 60)
    logger.info("LoRA Training: Kobudo/Karate Item Recognition")
    logger.info("=" * 60)
    logger.info(f"Base Model: {config.base_model}")
    logger.info(f"Data Path: {config.data_path}")
    logger.info(f"Output Dir: {config.output_dir}")
    logger.info("=" * 60)

    set_seed(config.seed)

    if config.smoke_test:
        logger.warning("!" * 60)
        logger.warning("SMOKE TEST MODE — minimal run to verify pipeline integrity")
        logger.warning("!" * 60)
        config.per_device_batch_size = 1
        config.gradient_accumulation_steps = 1
        config.max_seq_length = 512
        config.num_workers = 0
        config.output_dir = str(Path(config.output_dir).parent / "smoke_test")

    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config_dict = {
        k: v if not isinstance(v, list) else v for k, v in config.__dict__.items()
    }
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    model, processor = load_model_and_processor(config)

    # Create dataset
    image_folder = os.path.join(config.data_path, "images")
    if not os.path.exists(image_folder):
        image_folder = config.data_path

    logger.info(f"Loading images from: {image_folder}")
    full_dataset = KobudoDataset(
        image_folder, processor, max_length=config.max_seq_length
    )

    # Split train/eval 90/10
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    if config.smoke_test:
        indices = indices[: min(8, len(indices))]
    eval_size = max(1, min(len(indices) // 10, 50))
    train_size = len(indices) - eval_size

    train_dataset = Subset(full_dataset, indices[:train_size])
    eval_dataset = Subset(full_dataset, indices[train_size:])

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")

    data_collator = DataCollatorForVision2Seq(
        processor=processor, model=model, padding=True
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_dir=os.path.join(config.output_dir, "logs"),
        max_steps=5 if config.smoke_test else -1,
        logging_steps=1 if config.smoke_test else config.logging_steps,
        save_steps=config.save_steps if not config.smoke_test else 9999,
        eval_steps=2 if config.smoke_test else config.eval_steps,
        eval_strategy="steps",
        save_total_limit=0 if config.smoke_test else 3,
        load_best_model_at_end=not config.smoke_test,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(config.mixed_precision == "bf16"),
        fp16=(config.mixed_precision == "fp16"),
        dataloader_num_workers=config.num_workers,
        seed=config.seed,
        report_to="tensorboard",
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training complete!")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

    if config.smoke_test:
        logger.info("Smoke test passed — pipeline is healthy. No model saved.")
        return model, processor

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(config.output_dir, "final"))
    processor.save_pretrained(os.path.join(config.output_dir, "final"))

    summary = {
        "model": config.base_model,
        "data_path": config.data_path,
        "output_dir": config.output_dir,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "num_epochs": config.num_train_epochs,
        "lora_rank": config.lora_rank,
        "completed": True,
    }

    with open(os.path.join(config.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Done! Model saved to: {config.output_dir}")
    logger.info("=" * 60)

    return model, processor


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train LoRA for Kobudo/Karate Item Recognition (English only)"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="./training_data",
        help="Path to training data folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/kobudo_lora",
        help="Output directory",
    )
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a minimal 5-step pass to verify the pipeline before a full training run",
    )

    args = parser.parse_args()

    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        smoke_test=args.smoke_test,
    )

    train(config)


if __name__ == "__main__":
    main()
