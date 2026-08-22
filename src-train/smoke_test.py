#!/usr/bin/env python3
"""
Smoke test script for Axolotl fine-tuning environment.
Prints Axolotl version, PyTorch CUDA support, and detailed GPU hardware/VRAM statistics.
"""

import sys
import os
import platform
import importlib
import importlib.metadata

def get_pkg_version(pkg_name: str) -> str:
    """Helper to safely retrieve package version."""
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return "Not Installed"
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=" * 60)
    print("      Axolotl GPU Cluster Environment Smoke Test")
    print("=" * 60)

    # 1. System & Python Info
    print("\n[System Info]")
    print(f"  Python Version : {sys.version.split()[0]}")
    print(f"  Platform       : {platform.platform()}")

    # 2. Axolotl & Core ML Libraries
    print("\n[ML Core Libraries]")
    
    # Try importing axolotl dynamically to prevent local LSP missing import errors
    axolotl_ver = get_pkg_version("axolotl")
    if axolotl_ver == "Not Installed":
        try:
            axolotl = importlib.import_module("axolotl")
            axolotl_ver = getattr(axolotl, "__version__", "Installed (unknown version)")
        except ImportError:
            axolotl_ver = "Not Installed / Import Failed"

    print(f"  Axolotl        : {axolotl_ver}")
    print(f"  Transformers   : {get_pkg_version('transformers')}")
    print(f"  PEFT           : {get_pkg_version('peft')}")
    print(f"  Accelerate     : {get_pkg_version('accelerate')}")
    print(f"  BitsAndBytes   : {get_pkg_version('bitsandbytes')}")
    print(f"  Triton         : {get_pkg_version('triton')}")
    print(f"  LLMCompressor  : {get_pkg_version('llmcompressor')}")

    # Functional test for llmcompressor oneshot API
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        _ = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC")
        print("  LLMCompressor  : Functional (oneshot & QuantizationModifier verified)")
    except Exception as e:
        print(f"  LLMCompressor  : Functional Test Failed ({e})")

    # 3. PyTorch & CUDA Information
    print("\n[PyTorch & CUDA Environment]")
    try:
        torch = importlib.import_module("torch")
        print(f"  PyTorch        : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA Available : {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA Version   : {torch.version.cuda}")
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                print(f"  cuDNN Version  : {torch.backends.cudnn.version()}")
            
            device_count = torch.cuda.device_count()
            print(f"  GPU Count      : {device_count}")

            print("\n[GPU Details]")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = props.total_memory / (1024 ** 3)
                
                try:
                    free_mem_bytes, _ = torch.cuda.mem_get_info(i)
                    free_mem_gb = free_mem_bytes / (1024 ** 3)
                except Exception:
                    free_mem_gb = -1.0

                print(f"  --- GPU {i}: {device_name} ---")
                print(f"      Compute Capability : {cap[0]}.{cap[1]}")
                print(f"      SM Count           : {props.multi_processor_count}")
                print(f"      Total VRAM         : {total_mem_gb:.2f} GB")
                if free_mem_gb >= 0:
                    print(f"      Free VRAM          : {free_mem_gb:.2f} GB")
        else:
            print("  WARNING: CUDA is NOT available to PyTorch!")

    except ImportError:
        print("  ERROR: PyTorch is not installed in this environment.")

    print("=" * 60)

if __name__ == "__main__":
    main()
