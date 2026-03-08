"""
setup.py — Install all dependencies for ASFT training.
Run once before training:  python setup.py
"""

import subprocess
import sys


def run(cmd):
    print(f"▶ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def main():
    print("=" * 60)
    print("ASFT Environment Setup")
    print("=" * 60)

    # Uninstall any conflicting old versions
    print("\n[1/3] Removing old transformers / peft...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "transformers", "peft"],
        capture_output=True,
    )

    # Install core dependencies
    print("\n[2/3] Installing dependencies...")
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/huggingface/transformers",
        "accelerate>=0.26.0",
        "peft>=0.9.0",
        "bitsandbytes>=0.43.0",
        "qwen-vl-utils",
        "Pillow",
        "tqdm",
    ])

    # Verify imports
    print("\n[3/3] Verifying imports...")
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig, get_peft_model
        from qwen_vl_utils import process_vision_info
        print("\n✅ All imports OK — ready to train!")
        print("   Next step:  python train.py")
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   If running in a notebook, restart the kernel first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
