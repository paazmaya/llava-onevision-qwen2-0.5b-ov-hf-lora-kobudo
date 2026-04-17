#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Kobudo/Karate Item Recognition
Using PEFT with LLaVA-OneVision models

Optimized for: Windows 11, NVIDIA GTX 4070 (12GB VRAM)
Base Model: llava-hf/llava-onevision-qwen2-0.5b-ov-hf
Data Format: Folder-based images
Language: English only

Requirements:
    uv venv
    uv pip install -r requirements.txt
    python train_kobudo_lora.py

Author: MiniMax Agent
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for LoRA training - optimized for 12GB VRAM"""
    base_model: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    data_path: str = "./training_data"
    output_dir: str = "./output/kobudo_lora"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    mixed_precision: str = "bf16"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    seed: int = 42
    num_workers: int = 2

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
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


def auto_generate_caption(filename: str) -> str:
    """
    Generate English caption based on filename for kobudo/karate items.
    Specializes the model to recognize and describe Okinawan martial arts equipment.
    """
    filename_lower = filename.lower().replace('_', ' ').replace('-', ' ')

    # Kobudo weapons and equipment
    kobudo_items = {
        'tonfa': 'Traditional Okinawan tonfa, a wooden baton with a perpendicular handle used in kobudo martial arts for striking and blocking techniques',
        'sai': 'Sai, a traditional three-pronged Okinawan metal truncheon used in kobudo as a defensive and striking weapon',
        'bo': 'Bo staff, a six-foot wooden staff, the most fundamental weapon in Okinawan kobudo for developing strength and coordination',
        'nunchaku': 'Nunchaku, two wooden sticks connected by rope or chain, adapted from agricultural rice flails for karate and kobudo training',
        'kama': 'Kama, a traditional Okinawan sickle weapon with a curved blade used in kobudo for slashing and trapping techniques',
        'eiku': 'Eiku, an Okinawan boat oar converted into a weapon featuring a long handle with a flat paddle end',
        'tekko': 'Tekko, traditional Okinawan metal knuckles worn on the hand for striking in karate and kobudo',
        'timbe': 'Timbe, a short Okinawan staff with a curved end used in traditional kobudo',
        'surujin': 'Surujin, rope dart consisting of a length of rope with a dart attached for striking and entangling',
        'kuwa': 'Kuwa, a farming pitchfork adapted into a kobudo weapon with multiple prongs',
        'unku': 'Unku, an Okinawan staff weapon with a hook on one end',
        'rochin': 'Rochin, a short Okinawan spear with a leaf-shaped blade',
        'jutte': 'Jutte, a metal truncheon with a hook used in traditional martial arts',
        'tanbo': 'Tanbo, a short staff used in martial arts training',
        'sanshakubo': 'Sanshakubo, a three-foot wooden staff used in Okinawan kobudo',
        'bokken': 'Bokken, a wooden training sword used in karate and kobudo dojos',
        'shi': 'Shi, a short metal staff or rod weapon',
        'sansetsukon': 'Sansetsukon, a three-section staff with three sticks connected by rope',
        'jo': 'Jo staff, a four-foot short staff used in Japanese martial arts',
        'shaken': 'Shaken, a metal fan weapon used in some kobudo styles'
    }

    # Karate equipment
    karate_items = {
        'makiwara': 'Makiwara, a traditional karate striking board for developing focused striking power and technique',
        'bosu': 'Bosu, a balance training device used in martial arts for developing core strength and stability',
        'heavybag': 'Heavy bag for striking practice in karate training',
        'focusmitt': 'Focus mitts used by training partners for precision striking practice',
        'kick shield': 'Kick shield, a large padded target for kicking practice in karate',
        'double end bag': 'Double end bag, a small suspended ball for developing timing and reflexes',
        'gi': 'Karate gi, traditional training uniform made of heavy cotton for karate practice',
        'belt': 'Belt, colored rank indicator in karate representing skill level and experience',
        'punching bag': 'Punching bag for developing power and technique in karate'
    }

    all_items = {**kobudo_items, **karate_items}

    # Check for matches in filename
    for item_name, description in all_items.items():
        if item_name in filename_lower:
            return f"Describe this Okinawan martial arts item in detail. {description}"

    # Generic English-only caption
    return "Describe this Okinawan martial arts equipment in detail, including its name, type, traditional use, and any distinguishing features. Respond in English only."


from torch.utils.data import Dataset
import torch
from PIL import Image


class KobudoDataset(Dataset):
    """Dataset for kobudo/karate items from folder-based images"""

    def __init__(self, image_folder: str, processor: Any, max_length: int = 2048):
        self.image_folder = Path(image_folder)
        self.processor = processor
        self.max_length = max_length
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

        # Find all images
        self.image_files = []
        for ext in self.image_extensions:
            self.image_files.extend(self.image_folder.glob(f'*{ext}'))
            self.image_files.extend(self.image_folder.glob(f'*{ext.upper()}'))

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
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            image = Image.new('RGB', (336, 336), color='black')

        # Generate English caption
        caption = auto_generate_caption(image_path.stem)

        inputs = self.processor(
            text=[caption],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        return {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}


def load_model_and_processor(config: TrainingConfig):
    """Load model and processor with LoRA applied"""
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import LoraConfig, get_peft_model

    logger.info(f"Loading base model: {config.base_model}")

    processor = AutoProcessor.from_pretrained(
        config.base_model,
        trust_remote_code=True
    )

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

    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model, processor


def train(config: TrainingConfig):
    """Main training function"""
    from transformers import TrainingArguments, Trainer, DataCollatorForVision2Seq, set_seed
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
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config_dict = {k: v if not isinstance(v, list) else v for k, v in config.__dict__.items()}
    with open(os.path.join(config.output_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)

    model, processor = load_model_and_processor(config)

    # Create dataset
    image_folder = os.path.join(config.data_path, "images")
    if not os.path.exists(image_folder):
        image_folder = config.data_path

    logger.info(f"Loading images from: {image_folder}")
    full_dataset = KobudoDataset(image_folder, processor)

    # Split train/eval 90/10
    eval_size = min(len(full_dataset) // 10, 50)
    train_size = len(full_dataset) - eval_size
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    train_dataset = Subset(full_dataset, indices[:train_size])
    eval_dataset = Subset(full_dataset, indices[train_size:])

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")

    data_collator = DataCollatorForVision2Seq(processor=processor, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
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
        "completed": True
    }

    with open(os.path.join(config.output_dir, "training_summary.json"), 'w') as f:
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

    parser.add_argument("--data-path", type=str, default="./training_data",
                       help="Path to training data folder")
    parser.add_argument("--output-dir", type=str, default="./output/kobudo_lora",
                       help="Output directory")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr
    )

    train(config)


if __name__ == "__main__":
    main()
