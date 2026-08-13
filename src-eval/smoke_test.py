#!/usr/bin/env python3
"""
Smoke test script for vLLM evaluation environment.
Prints vLLM version, PyTorch CUDA support, and detailed GPU hardware/VRAM statistics.
"""

import importlib
import importlib.metadata
import platform
import sys

def get_pkg_version(pkg_name: str) -> str:
    """Helper to safely retrieve package version."""
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return "Not Installed"
    except (AttributeError, ValueError) as e:
        return f"Error: {e}"

def main() -> None:
    print("=" * 60)
    print("      vLLM Evaluation Cluster Environment Smoke Test")
    print("=" * 60)

    # 1. System & Python Info
    print("\n[System Info]")
    print(f"  Python Version : {sys.version.split()[0]}")
    print(f"  Platform       : {platform.platform()}")

    # 2. vLLM & Core Inference Libraries
    print("\n[Inference Core Libraries]")
    
    vllm_ver = get_pkg_version("vllm")
    if vllm_ver == "Not Installed":
        try:
            vllm = importlib.import_module("vllm")
            vllm_ver = str(getattr(vllm, "__version__", "Installed (unknown version)"))
        except ImportError:
            vllm_ver = "Not Installed / Import Failed"

    print(f"  vLLM           : {vllm_ver}")
    print(f"  Transformers   : {get_pkg_version('transformers')}")
    print(f"  Ray            : {get_pkg_version('ray')}")
    print(f"  TrtLLM         : {get_pkg_version('tensorrt_llm')}")

    # 3. PyTorch & CUDA Information
    print("\n[PyTorch & CUDA Environment]")
    try:
        torch = importlib.import_module("torch")
        print(f"  PyTorch        : {getattr(torch, '__version__', 'Unknown')}")
        cuda_available = bool(torch.cuda.is_available())
        print(f"  CUDA Available : {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA Version   : {getattr(torch.version, 'cuda', 'Unknown')}")
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                print(f"  cuDNN Version  : {torch.backends.cudnn.version()}")
            
            device_count = int(torch.cuda.device_count())
            print(f"  GPU Count      : {device_count}")

            print("\n[GPU Details]")
            for i in range(device_count):
                device_name = str(torch.cuda.get_device_name(i))
                cap = torch.cuda.get_device_capability(i)
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = float(props.total_memory / (1024 ** 3))
                
                try:
                    free_mem_bytes, _ = torch.cuda.mem_get_info(i)
                    free_mem_gb = float(free_mem_bytes / (1024 ** 3))
                except (AttributeError, RuntimeError):
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
