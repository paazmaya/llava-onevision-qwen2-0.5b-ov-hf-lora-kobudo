#!/usr/bin/env python3
"""
Interactive Inference Script for Kobudo/Karate Item Recognition
Using a LoRA adapter fine-tuned on top of LLaVA-OneVision

Optimized for: Windows 11, NVIDIA RTX 4070 (12GB VRAM)
Base Model: llava-hf/llava-onevision-qwen2-0.5b-ov-hf

Usage:
    uv run python infer_kobudo.py --lora-dir ./output/kobudo_lora/final
"""

import os
import argparse
import logging

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def interactive_inference(
    base_model: str,
    lora_dir: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    from peft import PeftModel
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

    logger.info("Loading base model...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    if os.path.exists(lora_dir):
        model = PeftModel.from_pretrained(model, lora_dir)
        logger.info(f"Loaded LoRA adapter from {lora_dir}")
    else:
        logger.warning(
            f"LoRA directory not found: {lora_dir}. Running base model only."
        )

    model.eval()

    print("\n" + "=" * 60)
    print("Interactive Inference Mode")
    print("=" * 60)
    print("Enter image path (or 'quit' to exit)\n")

    while True:
        try:
            image_path = input("Image path: ").strip()

            if image_path.lower() in ["quit", "exit", "q"]:
                break

            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "You are an expert in Okinawan martial arts equipment. "
                                "Describe this item in detail, including its name, type, "
                                "traditional use, and any distinguishing features. "
                                "Respond in English only."
                            ),
                        },
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            inputs = processor(text=[prompt], images=[image], return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )

            response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            print("\n" + "-" * 40)
            print("RESPONSE:")
            print(response)
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive inference for Kobudo/Karate item recognition"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        help="HuggingFace model ID or local path of the base model",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default="./output/kobudo_lora/final",
        help="Path to the saved LoRA adapter directory",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling probability"
    )

    args = parser.parse_args()

    interactive_inference(
        base_model=args.base_model,
        lora_dir=args.lora_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
