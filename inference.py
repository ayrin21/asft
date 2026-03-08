"""
inference.py — Run ASFT fine-tuned Qwen2.5-VL-3B on turbulence-blurred book covers.

Usage:
    python inference.py --model_dir ./outputs --image path/to/image.jpg
    python inference.py --model_dir ./outputs --image path/to/image.jpg --no_quant
"""

import argparse
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info


SYSTEM_PROMPT = (
    "You are an expert OCR system specialised in reading text from "
    "book covers that may be degraded by atmospheric turbulence blur. "
    "Extract ALL visible text exactly as it appears."
)


def load_model(model_dir: str, use_quant: bool = True):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if use_quant else None

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    return model, processor


def predict(model, processor, image_path: str, max_new_tokens: int = 256) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text from this book cover."},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the new tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="ASFT inference on book covers")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to fine-tuned model directory")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--no_quant", action="store_true",
                        help="Disable 4-bit quantisation (higher VRAM)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    model, processor = load_model(args.model_dir, use_quant=not args.no_quant)

    print(f"Running inference on {args.image}...")
    result = predict(model, processor, args.image, args.max_new_tokens)

    print("\n" + "=" * 50)
    print("Extracted Text:")
    print("=" * 50)
    print(result)


if __name__ == "__main__":
    main()
