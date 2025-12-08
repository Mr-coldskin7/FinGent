import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.get_device_name(0))