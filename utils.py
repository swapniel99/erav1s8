import torch

SEED = 42


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Device Selected:", device)

    return device


def set_seed(device=None, seed=SEED):
    if device is None:
        device = get_device()

    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
