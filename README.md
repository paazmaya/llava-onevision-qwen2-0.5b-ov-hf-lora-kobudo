# llava-onevision-qwen2-0.5b-ov-hf-lora-kobudo

LoRA fine-tuning of [llava-hf/llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) for recognizing and describing Okinawan martial arts equipment (kobudo weapons and karate gear).

## What it does

The script fine-tunes the LLaVA-OneVision vision-language model using Low-Rank Adaptation (LoRA) so that it can identify and describe kobudo weapons (tonfa, sai, bo, nunchaku, kama, etc.) and karate equipment (makiwara, gi, heavy bag, etc.) from images. Responses are English-only.

Two modes are available:

- **train** — Fine-tunes the model on a local folder of images and saves the LoRA adapter.
- **interactive** — Available as a separate script (`infer_kobudo.py`). Loads the saved adapter and lets you query it interactively with image paths.

### Training details

- Base model: `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
- Optimized for a single GPU with 12 GB VRAM (e.g. NVIDIA RTX 4070)
- LoRA rank 16, alpha 32, targeting attention and MLP projection layers
- Mixed precision: bfloat16
- Batch size 1 with 8 gradient accumulation steps (effective batch 8)
- 90 / 10 train / eval split (eval capped at 50 samples)
- Checkpoints and TensorBoard logs written to `./output/kobudo_lora/`

## Requirements

- Python 3.13+
- CUDA-capable GPU (12 GB VRAM recommended)
- [uv](https://github.com/astral-sh/uv)

Install dependencies:

```bash
uv venv
uv sync
```

## Training data

Place your images inside a folder. The script looks for images first in `<data-path>/images/`, falling back to `<data-path>/` directly. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

Captions are generated automatically from filenames. If the filename contains a known weapon or equipment name (e.g. `tonfa_01.jpg`) the model receives a detailed description of that item as its training prompt. Unknown filenames receive a generic prompt.

Default data path: `./training_data`

## How to run

### Train

```bash
python train_kobudo_lora.py --mode train --data-path ./training_data --output-dir ./output/kobudo_lora
```

All arguments and their defaults:

| Argument | Default | Description |
|---|---|---|
| `--data-path` | `./training_data` | Path to the image folder |
| `--output-dir` | `./output/kobudo_lora` | Where to save checkpoints and the final adapter |
| `--rank` | `16` | LoRA rank |
| `--alpha` | `32` | LoRA alpha |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `1` | Per-device batch size |
| `--lr` | `1e-4` | Learning rate |

### Interactive inference

```bash
python infer_kobudo.py --lora-dir ./output/kobudo_lora/final
```

All arguments and their defaults:

| Argument | Default | Description |
|---|---|---|
| `--base-model` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | Base model ID or local path |
| `--lora-dir` | `./output/kobudo_lora/final` | Path to the saved LoRA adapter |
| `--max-new-tokens` | `512` | Maximum tokens to generate |
| `--temperature` | `0.1` | Sampling temperature |
| `--top-p` | `0.9` | Top-p sampling probability |

You will be prompted to enter an image path. Type `quit` to exit.

## Output

After training the following files are written to `--output-dir`:

- `config.json` — training configuration snapshot
- `training_summary.json` — final statistics (sample counts, epochs, etc.)
- `final/` — the saved LoRA adapter and processor, ready for `PeftModel.from_pretrained`
- `logs/` — TensorBoard event files
