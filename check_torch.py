import torch
import torchaudio
import torchvision
import transformers

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Torchaudio version:", torchaudio.__version__)
print("Transformers version:", transformers.__version__)
print("CUDA/ROCm available:", torch.cuda.is_available())
if hasattr(torch.version, "hip"):
    print("ROCm version:", torch.version.hip)
else:
    print("ROCm version: N/A")

print("\nRunning compute test...")
if torch.cuda.is_available():
    # Test actual GPU compute
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.mm(x, x)
    print("✓ Matrix multiply test passed!")
    del x, y
    torch.cuda.synchronize()

if torch.cuda.is_available():
    print("\nGPU Information:")
    print("GPU Count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
        props = torch.cuda.get_device_properties(i)
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(
            f"  - GCN Arch: gfx{props.gcnArchName}"
            if hasattr(props, "gcnArchName")
            else ""
        )
