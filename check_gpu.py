import torch

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current device index: {torch.cuda.current_device()}")
    else:
        print("GPU is NOT available. Using CPU.")

if __name__ == "__main__":
    check_gpu()