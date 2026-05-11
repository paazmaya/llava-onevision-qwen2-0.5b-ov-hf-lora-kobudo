#!/usr/bin/env python3
"""
Describe images using a LLaVA-OneVision model (7B recommended).

Accepts either a GGUF file or a HuggingFace model folder (safetensors).

For each image found in the given folder a sidecar .txt file is written that
covers:
  - Environment / setting
  - Each person separately: apparent gender, outfit, objects held
  - Placeholder [WEAPON: TBD] for martial-arts weapons that cannot yet be
    identified — fill this in manually after the script finishes.

Usage (GGUF):
    uv run python pre_describe_images.py --model /path/to/model.gguf --image-dir /path/to/images
    uv run python pre_describe_images.py --model /path/to/model.gguf \\
                                         --mmproj /path/to/mmproj.gguf \\
                                         --image-dir /path/to/images \\
                                         --overwrite

Usage (HuggingFace folder):
    uv run python pre_describe_images.py --model /path/to/hf-model-folder --image-dir /path/to/images
    uv run python pre_describe_images.py --model /path/to/hf-model-folder \\
                                         --image-dir /path/to/images \\
                                         --overwrite
"""

import argparse
import base64
import logging
import mimetypes
from pathlib import Path
from typing import Callable

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".heic",
    ".heif",
}

SYSTEM_PROMPT = (
    "You are an expert image analyst helping to build training data for an Okinawan "
    "kobudo (traditional weapons martial arts) image recognition system. "
    "Describe only what is directly visible. Never infer, guess, or embellish. "
    "If you are not certain about a detail, omit it entirely rather than speculate."
)

USER_PROMPT = (
    "Analyse this image and respond in the structured format below. "
    "Use English only. Only describe what you can directly see — do not infer, guess, "
    "or add details that are not clearly visible.\n\n"
    "ENVIRONMENT:\n"
    "Describe the setting: indoor or outdoor, floor/ground surface, "
    "background elements, lighting conditions, and any visible props or surroundings. "
    "Only include details you can clearly see.\n\n"
    "PEOPLE:\n"
    "List every person visible in the image as a numbered entry. "
    "For each person state:\n"
    "  - Gender: state male or female based solely on visible appearance. "
    "Do not use the word 'apparent' or hedge in any way.\n"
    "  - Outfit and clothing: state only the colours and garment types you can clearly see "
    "(e.g. karategi, hakama, dogi, casual). Do not add details such as trim or patterns "
    "unless they are unambiguously visible.\n"
    "  - Objects held or worn as equipment: name only objects you can clearly identify. "
    "    * If a person holds or wears a martial-arts weapon or training tool that you "
    "cannot clearly and confidently name, write exactly this placeholder on its own line:\n"
    "      [WEAPON: TBD]\n"
    "    * Do not guess or suggest what an object might be.\n"
    "    * If they are not holding anything, state that explicitly.\n\n"
    "If no people appear in the image, state nothing about people."
)


def _mime_type(image_path: Path) -> str:
    """Return a safe MIME type string for the image."""
    guessed, _ = mimetypes.guess_type(image_path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    # Fallback based on suffix
    fallback = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".gif": "image/gif",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }
    return fallback.get(image_path.suffix.lower(), "image/jpeg")


def _image_data_uri(image_path: Path) -> str:
    """Encode an image as a base-64 data URI."""
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{_mime_type(image_path)};base64,{b64}"


def _describe_image_gguf(llm, image_path: Path, max_tokens: int) -> str:
    """Run a llama-cpp GGUF model on one image and return the description."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_uri(image_path)},
                },
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    response = llm.create_chat_completion(messages=messages, max_tokens=max_tokens)
    return response["choices"][0]["message"]["content"].strip()


def _describe_image_hf(model, processor, image_path: Path, max_tokens: int) -> str:
    """Run a HuggingFace transformers model on one image and return the description."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(
        images=image, text=prompt, return_tensors="pt"
    ).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def process_folder(
    infer_fn: Callable[[Path, int], str],
    image_dir: Path,
    max_tokens: int,
    overwrite: bool,
) -> None:
    all_images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(image_dir.rglob(f"*{ext}"))
        all_images.extend(image_dir.rglob(f"*{ext.upper()}"))
    image_files = sorted(set(all_images))

    if not image_files:
        logger.warning("No supported images found in %s", image_dir)
        return

    logger.info("Found %d image(s) in %s", len(image_files), image_dir)

    skipped = 0
    for image_path in image_files:
        txt_path = image_path.with_suffix(".txt")

        if txt_path.exists() and not overwrite:
            logger.info(
                "Skipping %s (sidecar exists; use --overwrite to replace)",
                image_path.name,
            )
            skipped += 1
            continue

        logger.info("Describing %s ...", image_path.name)
        try:
            description = infer_fn(image_path, max_tokens)
            txt_path.write_text(description, encoding="utf-8")
            logger.info("  Saved -> %s", txt_path.name)
        except Exception as exc:
            logger.error("Failed to process %s: %s", image_path.name, exc)

    if skipped:
        logger.info(
            "%d file(s) skipped. Run with --overwrite to regenerate them.", skipped
        )


def _find_mmproj(model_path: Path) -> str | None:
    """Try to locate an mmproj GGUF file next to the model."""
    candidates = sorted(model_path.parent.glob("mmproj*.gguf"))
    return str(candidates[0]) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use a LLaVA-OneVision model to describe people and environments "
            "in images and save the descriptions as sidecar .txt files. "
            "Accepts either a GGUF file or a HuggingFace model folder."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help=(
            "Path to a GGUF model file (e.g. llava-onevision-qwen2-7b-ov-hf_Q4_K_M.gguf) "
            "OR a folder containing HuggingFace safetensors model files."
        ),
    )
    parser.add_argument(
        "--mmproj",
        default=None,
        metavar="MMPROJ_GGUF",
        help=(
            "Path to the multimodal projector GGUF file (mmproj-*.gguf). "
            "Only used with GGUF models. If omitted the script looks for a file "
            "matching mmproj*.gguf in the same directory as --model."
        ),
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        metavar="DIR",
        help="Directory containing the images to describe.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per image (default: 512).",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="(GGUF only) Layers to offload to GPU. -1 = all layers (default), 0 = CPU only.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="(GGUF only) Context window size (default: 4096).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sidecar .txt files.",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    image_dir = Path(args.image_dir).resolve()

    if not (model_path.is_file() or model_path.is_dir()):
        parser.error(f"Model path not found: {model_path}")
    if not image_dir.is_dir():
        parser.error(f"Image directory not found: {image_dir}")

    if model_path.is_dir():
        # HuggingFace safetensors folder
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

        logger.info("Loading HuggingFace model from folder: %s", model_path)
        processor = AutoProcessor.from_pretrained(str(model_path))
        hf_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            str(model_path), device_map="auto"
        )
        hf_model.eval()
        logger.info("HuggingFace model loaded.")

        def infer_fn(image_path: Path, max_tokens: int) -> str:
            return _describe_image_hf(hf_model, processor, image_path, max_tokens)

    else:
        # GGUF file
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        mmproj_path = args.mmproj
        if mmproj_path is None:
            mmproj_path = _find_mmproj(model_path)
            if mmproj_path:
                logger.info("Auto-detected mmproj: %s", mmproj_path)
            else:
                parser.error(
                    "No mmproj GGUF file found next to the model. "
                    "Pass it explicitly with --mmproj <path>."
                )

        logger.info("Loading GGUF model: %s", model_path)
        chat_handler = Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False,
        )
        llm = Llama(
            model_path=str(model_path),
            chat_handler=chat_handler,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False,
        )
        logger.info("GGUF model loaded.")

        def infer_fn(image_path: Path, max_tokens: int) -> str:
            return _describe_image_gguf(llm, image_path, max_tokens)

    process_folder(infer_fn, image_dir, args.max_tokens, args.overwrite)
    logger.info("Done.")


if __name__ == "__main__":
    main()
