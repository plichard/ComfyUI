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
