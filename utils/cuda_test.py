import torch

if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available")
