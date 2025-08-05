import torch

def is_linux():
    from sys import platform
    if platform == "linux" or platform == "linux2":
        return True
    return False

if __name__ == "__main__":
    device = is_linux()
    print(f"Is linux: {device}")