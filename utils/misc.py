# Miscellaneous helper functions
import torch
from importlib.metadata import version

def print_lib_versions():
    print(f"{50 * '='}")
    print("\t\tPACKAGES VERSIONS")
    print(f"{50 * '='}")
    pkgs = ["matplotlib", 
        "numpy", 
        "tiktoken", 
        "torch"
       ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

# Helper function to get the default device based on availability
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def set_device(device_name=None):
    # The following lines will allow the code to run on Apple Silicon chips, if applicable,
    # which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
    # However, the resulting loss values may be slightly different.

    if device_name is not None:
        if device_name == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_name == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif device_name == "cpu":
            device = torch.device("cpu")
        else:
            print(f"Warning: {device_name} device not available")
            device = get_default_device()
    else:
        device = get_default_device()
    return device