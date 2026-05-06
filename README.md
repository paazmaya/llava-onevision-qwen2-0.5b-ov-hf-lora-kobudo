# llava-onevision-qwen2-0.5b-ov-hf-lora-kobudo

LoRA fine-tuning of [llava-hf/llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) for recognizing and describing Okinawan martial arts equipment (kobudo weapons and karate gear).

[Read about LLaVA-OneVision - Easy Visual Task Transfer at their blog](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)

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

Now that the config is in 4-bit, the 7B model should fit in my 12 GB vram...

```powershell
uv run python train_kobudo_lora.py --base-model "H:\vision-models\llava-onevision-qwen2-7b-ov-hf" --data-path "C:\Users\Jukka\Dropbox\onevision-lora-kobudo-images" --output-dir ./output/kobudo_lora
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
```bash
uv run python infer_kobudo.py --lora-dir ./output/kobudo_lora_7b/final --base-model "H:\vision-models\llava-onevision-qwen2-7b-ov-hf"
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

## Getting smaller base model

LLaVA-OneVision is a multimodal model — converting it to GGUF requires two
separate files: a backbone GGUF and a vision-projector (mmproj) GGUF.
The Python helper scripts (`llava_surgery_v2.py`,
`convert_image_encoder_to_gguf.py`) are **not bundled** in the Docker image,
so Steps 1 and 3 must be run locally from a cloned llama.cpp repo.

```powershell
# One-time clone
git clone https://github.com/ggml-org/llama.cpp.git C:\Users\Jukka\code\github-other\llama.cpp
pip install torch safetensors
pip install C:\Users\Jukka\code\github-other\llama.cpp\gguf-py

# Pull the CUDA-enabled image (needed for Steps 2 and 4)
docker pull ghcr.io/ggml-org/llama.cpp:full-cuda13

# Step 1 — split out the vision projector and patch the model directory.
# Writes llava.projector + llava.clip alongside the HF weights and removes
# tensors that convert_hf_to_gguf.py cannot handle (image_newline, etc.).
python C:\Users\Jukka\code\github-other\llama.cpp\tools\mtmd\legacy-models\llava_surgery_v2.py `
    -C -m "H:\vision-models\llava-onevision-qwen2-7b-ov-hf"

# Step 1b — llava_surgery_v2.py extracts image_newline into llava.projector but
# does NOT remove it from the safetensors shards or the index. Do that manually:
python -c "
from safetensors import safe_open
from safetensors.torch import save_file
from typing import cast, Any, ContextManager
import glob, json, os

model_path = r'H:\vision-models\llava-onevision-qwen2-7b-ov-hf'

# Remove from shard
for path in [f for f in glob.glob(f'{model_path}/*.safetensors') if 'model' in os.path.basename(f)]:
    tensors = {}
    with cast(ContextManager[Any], safe_open(path, framework='pt', device='cpu')) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).clone()
    if 'image_newline' in tensors:
        del tensors['image_newline']
        save_file(tensors, path)
        print(f'Removed image_newline from {os.path.basename(path)}')

# Remove from index
idx_path = f'{model_path}/model.safetensors.index.json'
with open(idx_path) as f:
    idx = json.load(f)
idx['weight_map'].pop('image_newline', None)
with open(idx_path, 'w') as f:
    json.dump(idx, f, indent=2)
print('Index updated.')
"

# Step 2 — convert the patched LLM backbone to FP16 GGUF.
# tools.sh is the default entrypoint; --convert delegates to convert_hf_to_gguf.py.
docker run --gpus all -v "H:\vision-models\:/models" ghcr.io/ggml-org/llama.cpp:full-cuda13 `
    --convert /models/llava-onevision-qwen2-7b-ov-hf `
    --outtype f16 `
    --outfile /models/llava-onevision-qwen2-7b-ov-hf_f16.gguf

# Step 3 — convert the vision projector to its own mmproj GGUF.
# LLaVA-OneVision uses SigLIP which omits several keys the legacy script requires.
# Add them from known SigLIP defaults and from the text_config:
python -c "
import json
path = r'H:\vision-models\llava-onevision-qwen2-7b-ov-hf\config.json'
with open(path) as f: cfg = json.load(f)
cfg['projection_dim'] = cfg['text_config']['hidden_size']  # 3584
cfg['vision_config']['layer_norm_eps'] = 1e-6              # SigLIP default
cfg['vision_config']['hidden_act'] = 'gelu'                # SigLIP activation
with open(path, 'w') as f: json.dump(cfg, f, indent=2)
print('Config patched.')
"

python C:\Users\Jukka\code\github-other\llama.cpp\tools\mtmd\legacy-models\convert_image_encoder_to_gguf.py `
    --llava-projector "H:\vision-models\llava-onevision-qwen2-7b-ov-hf\llava.projector" `
    --output-dir "H:\vision-models" `
    -m "H:\vision-models\llava-onevision-qwen2-7b-ov-hf" `
    --clip-model-is-openclip
# Done. Output file: H:\vision-models\mmproj-model-f16.gguf

# Step 4 — quantize the FP16 backbone to the most useful sizes.
# --quantize delegates to llama-quantize inside the container.
docker run --gpus all -v "H:\vision-models\:/models" ghcr.io/ggml-org/llama.cpp:full-cuda13 `
    --quantize /models/llava-onevision-qwen2-7b-ov-hf_f16.gguf `
    /models/llava-onevision-qwen2-7b-ov-hf_Q4_K_M.gguf Q4_K_M
# llama_model_quantize_impl: model size  = 14527.15 MiB (16.00 BPW)
# llama_model_quantize_impl: quant size  =  4460.75 MiB (4.91 BPW)
# main: quantize time = 153863.51 ms
# main:    total time = 153863.51 ms

docker run --gpus all -v "H:\vision-models\:/models" ghcr.io/ggml-org/llama.cpp:full-cuda13 `
    --quantize /models/llava-onevision-qwen2-7b-ov-hf_f16.gguf `
    /models/llava-onevision-qwen2-7b-ov-hf_Q8_0.gguf Q8_0
# llama_model_quantize_impl: model size  = 14527.15 MiB (16.00 BPW)
# llama_model_quantize_impl: quant size  =  7718.14 MiB (8.50 BPW)
# main: quantize time = 181988.42 ms
# main:    total time = 181988.42 ms

docker run --gpus all -v "H:\vision-models\:/models" ghcr.io/ggml-org/llama.cpp:full-cuda13 `
    --quantize /models/llava-onevision-qwen2-7b-ov-hf_f16.gguf `
    /models/llava-onevision-qwen2-7b-ov-hf_Q2_K.gguf Q2_K
# llama_model_quantize_impl: model size  = 14527.15 MiB (16.00 BPW)
# llama_model_quantize_impl: quant size  =  2870.80 MiB (3.16 BPW)
# main: quantize time = 147674.36 ms
# main:    total time = 147674.36 ms
```

## Convert lora to gguf and test it

```powershell
# Convert the PEFT adapter to a GGUF lora file
# convert_lora_to_gguf.py is bundled in the Docker image
docker run --gpus all `
    -v "H:\vision-models\:/models" `
    -v "C:\Users\Jukka\code\github-paazmaya\mine\llava-onevision-qwen2-0.5b-ov-hf-lora-kobudo\output\kobudo_lora_7b\final:/lora" `
    --entrypoint python3 ghcr.io/ggml-org/llama.cpp:full-cuda13 `
    /app/convert_lora_to_gguf.py `
    --base /models/llava-onevision-qwen2-7b-ov-hf `
    --outfile /models/llava-onevision-qwen2-7b-ov-hf-kobudo-lora.gguf `
    /lora

# Then run inference with --lora
docker run --gpus all -v "H:\vision-models\:/models" --entrypoint /app/llama-mtmd-cli `
    ghcr.io/ggml-org/llama.cpp:full-cuda13 `
    -m /models/llava-onevision-qwen2-7b-ov-hf_Q4_K_M.gguf `
    --mmproj /models/mmproj-llava-onevision-qwen2-7b-ov-hf-f16.gguf `
    --lora /models/llava-onevision-qwen2-7b-ov-hf-kobudo-lora.gguf `
    --image /models/test.jpg `
    -p "What is this Okinawan martial arts item?"
```

### Running inference with llama-mtmd-cli

`llama-mtmd-cli` is the modern multimodal inference tool built into the image.
It requires both the backbone GGUF (`-m`) and the mmproj GGUF (`--mmproj`).

```powershell
# Interactive chat mode (no --image or -p)
docker run --gpus all -v "H:\vision-models\:/models" --entrypoint /app/llama-mtmd-cli `
    ghcr.io/ggml-org/llama.cpp:full-cuda13 `
    -m /models/llava-onevision-qwen2-7b-ov-hf_Q4_K_M.gguf `
    --mmproj /models/mmproj-llava-onevision-qwen2-7b-ov-hf-f16.gguf

# Single image query
docker run --gpus all -v "H:\vision-models\:/models" --entrypoint /app/llama-mtmd-cli `
    ghcr.io/ggml-org/llama.cpp:full-cuda13 `
    -m /models/llava-onevision-qwen2-7b-ov-hf_Q4_K_M.gguf `
    --mmproj /models/mmproj-llava-onevision-qwen2-7b-ov-hf-f16.gguf `
    --image /models/test.jpg `
    -p "What is this Okinawan martial arts item?"
```

## License

Apache-2.0
