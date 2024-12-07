import torch

def enable_cuda(no_cuda):
    """
    USING_PUNEET_GPU = 2
    USING_COLLAB_GPU = 3
    gpu_index = USING_COLLAB_GPU
    if "NVIDIA GeForce RTX 4050 Laptop GPU" == torch.cuda.get_device_name(0):
        gpu_index = USING_PUNEET_GPU
    print(f'GPU to be used: {"Collab Nvidia T4" if gpu_index == 1 else "Puneet RTX 4050"}')
    """
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("should enbale GPU? ", use_cuda)
    return device

