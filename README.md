# llava-onevision-qwen2-0.5b-ov-hf-lora-kobudo

LoRA fine-tuning of [llava-hf/llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) for recognizing and describing Okinawan martial arts equipment (kobudo weapons and karate gear).

## What it does

The script fine-tunes the LLaVA-OneVision vision-language model using Low-Rank Adaptation (LoRA) so that it can identify and describe kobudo weapons (tonfa, sai, bo, nunchaku, kama, etc.) and karate equipment (makiwara, gi, heavy bag, etc.) from images. Responses are English-only.

Two scripts are available:

- **train** — Fine-tunes the model on a local folder of images and saves the LoRA adapter.
- **inference** — Available as a separate script (`infer_kobudo.py`). Loads the saved adapter and lets you query it interactively with image paths.

### Training details

- Base model: `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
- Optimized for a single GPU with 12 GB VRAM (e.g. NVIDIA RTX 4070)
- LoRA rank 32, alpha 32, targeting attention and MLP projection layers
- Mixed precision: bfloat16
- Batch size 2 with 8 gradient accumulation steps (effective batch 16)
- Gradient checkpointing enabled to reduce activation memory
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

It is a good idea to pull the model locally:

```sh
hf download llava-hf/llava-onevision-qwen2-0.5b-ov-hf --local-dir "H:\vision-models\llava-onevision-qwen2-0.5b-ov-hf"

hf download llava-hf/llava-onevision-qwen2-7b-ov-hf --local-dir "H:\vision-models\llava-onevision-qwen2-7b-ov-hf"
```

## Training data

Place your images inside a folder. The script looks for images first in `<data-path>/images/`, falling back to `<data-path>/` directly. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

Captions are generated automatically from filenames. If the filename starts with a known weapon or equipment name followed by underscore (e.g. `tonfa_01.jpg`) the model receives a detailed description of that item as its training prompt. Unknown filenames receive a generic prompt.

Default data path: `./training_data`

```
training_data/
└── images/
    ├── tonfa_001.jpg
    ├── sai_weapon.jpg
    ├── bo_staff.png
    └── ...
```

## How to run

### Train

```bash
uv run python train_kobudo_lora.py --data-path ./training_data --output-dir ./output/kobudo_lora
```

```powershell
uv run python train_kobudo_lora.py --base-model "H:\vision-models\llava-hf_llava-onevision-qwen2-0.5b-ov-hf" --data-path "C:\Users\Jukka\Dropbox\onevision-lora-kobudo-images" --output-dir ./output/kobudo_lora
```




All arguments and their defaults:

| Argument           | Default                                     | Description                                                                        |
| ------------------ | ------------------------------------------- | ---------------------------------------------------------------------------------- |
| `--base-model`     | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | HuggingFace model ID or local path to the base model                               |
| `--data-path`      | `./training_data`                           | Path to the image folder (checked for an `images/` sub-folder first)               |
| `--output-dir`     | `./output/kobudo_lora`                      | Where to save checkpoints and the final adapter                                    |
| `--rank`           | `32`                                        | LoRA rank — higher values increase capacity but use more VRAM                      |
| `--alpha`          | `32`                                        | LoRA alpha — scaling factor, typically kept equal to `--rank`                      |
| `--epochs`         | `3`                                         | Number of full passes over the training data                                       |
| `--batch-size`     | `2`                                         | Per-device batch size (effective batch = batch-size × gradient-accumulation steps) |
| `--lr`             | `1e-5`                                      | Peak learning rate (cosine schedule with linear warmup)                            |
| `--max-image-size` | `768`                                       | Cap the longest image edge before AnyRes tiling (see note below)                   |
| `--smoke-test`     | `false`                                     | Run a minimal 5-step pass to verify the pipeline without a full training run       |

#### AnyRes tiling and `--max-image-size`

LLaVA-OneVision splits each image into 384 × 384 pixel tiles plus a thumbnail using the AnyRes algorithm. The number of tiles is determined by the nearest supported grid size (pinpoint). Larger images produce more tiles and therefore more vision tokens, which increases VRAM usage:

| `--max-image-size` | Max pinpoint fit | Tiles + thumbnail | ~Vision tokens |
| ------------------ | ---------------- | ----------------- | -------------- |
| `384`              | 384 × 384        | 1 + 1 = 2         | ~590           |
| `768` _(default)_  | 768 × 768        | 4 + 1 = 5         | ~820           |
| `1152`             | 1152 × 1152      | 9 + 1 = 10        | ~1960          |
| `1536`             | 1536 × 1536      | 16 + 1 = 17       | ~3300          |

Values that do not align with a pinpoint (e.g. `1024`) are silently upscaled to the next pinpoint by the processor, so tile counts do not decrease — use `384`, `768`, `1152`, or `1536` only.

### Interactive inference

```bash
uv run python infer_kobudo.py --lora-dir ./output/kobudo_lora/final
```

All arguments and their defaults:

| Argument           | Default                                     | Description                    |
| ------------------ | ------------------------------------------- | ------------------------------ |
| `--base-model`     | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | Base model ID or local path    |
| `--lora-dir`       | `./output/kobudo_lora/final`                | Path to the saved LoRA adapter |
| `--max-new-tokens` | `512`                                       | Maximum tokens to generate     |
| `--temperature`    | `0.1`                                       | Sampling temperature           |
| `--top-p`          | `0.9`                                       | Top-p sampling probability     |

You will be prompted to enter an image path. Type `quit` to exit.

## Output

After training the following files are written to `--output-dir`:

- `config.json` — training configuration snapshot
- `training_summary.json` — final statistics (sample counts, epochs, etc.)
- `final/` — the saved LoRA adapter and processor, ready for `PeftModel.from_pretrained`
- `logs/` — TensorBoard event files

## License

Apache-2.0
