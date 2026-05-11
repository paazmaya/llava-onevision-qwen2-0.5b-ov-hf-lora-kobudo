import time
import torch
from PIL import Image
from transformers import (
    LlavaOnevisionForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
LORA_PATH = "./kobudo-lora"  # your trained adapter
IMAGE_PATH = "test_weapon.jpg"

# llama.cpp files (from your earlier quantization)
GGUF_2BIT = "./llava-onevision-qwen2-7b-ov-hf_Q2_K.gguf"
MMPROJ_PATH = "./mmproj-llava-onevision-qwen2-7b-ov-hf-f16.gguf"  # if needed
LORA_GGML = "./kobudo-lora.ggml.bin"
N_CTX = 2048  # context length


# ------------------------------------------------------------
# 1. Inference using transformers: 8-bit base + LoRA
# ------------------------------------------------------------
def run_transformers_quant(quant_type):
    print(f"\n=== Testing {quant_type}-bit base + LoRA ===")
    if quant_type == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    image = Image.open(IMAGE_PATH).convert("RGB")
    prompt = "USER: <image>\nDescribe the weapon and its usage.\nASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    elapsed = time.time() - start

    result = processor.decode(output[0], skip_special_tokens=True)
    print(f"Output ({quant_type}-bit):\n{result}")
    print(f"Inference time: {elapsed:.2f}s")

    # Clean up to free GPU memory
    del model
    torch.cuda.empty_cache()


# ------------------------------------------------------------
# 2. Inference using llama.cpp: 2-bit base + LoRA
# ------------------------------------------------------------
def run_llamacpp_2bit():
    print("\n=== Testing 2-bit (Q2_K) base + LoRA via llama.cpp ===")

    # Create a chat handler that knows how to embed images for LLaVA
    chat_handler = Qwen25VLChatHandler(
        clip_model_path=MMPROJ_PATH,  # mmproj file if needed
        verbose=False,
    )

    llm = Llama(
        model_path=GGUF_2BIT,
        chat_handler=chat_handler,
        n_ctx=N_CTX,
        n_gpu_layers=-1,  # offload all layers to GPU if you want
        lora_path=LORA_GGML,  # apply your LoRA
        verbose=False,
    )

    # Prepare the chat message with the image
    image_uri = "file://" + IMAGE_PATH
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": "Describe the weapon and its usage."},
            ],
        }
    ]

    start = time.time()
    response = llm.create_chat_completion(messages=messages, max_tokens=128)
    elapsed = time.time() - start

    output = response["choices"][0]["message"]["content"]
    print(f"Output (2-bit):\n{output}")
    print(f"Inference time: {elapsed:.2f}s")

    del llm


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Ensure image exists
    Image.open(IMAGE_PATH)

    # 8‑bit
    run_transformers_quant(quant_type=8)

    # 4‑bit
    run_transformers_quant(quant_type=4)

    # 2‑bit (requires GGUF + converted LoRA)
    run_llamacpp_2bit()
