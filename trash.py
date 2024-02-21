import torch

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}, Compute Capability: {torch.cuda.get_device_capability(i)}")
else:
    print("CUDA is not available. Check your GPU drivers and CUDA installation.")
import torch

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
