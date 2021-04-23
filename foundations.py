import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cuda = torch.device('cuda')
    torch.cuda.current_device()
    torch.cuda.init()
    t1 = torch.rand(3, 3, device=cuda)
    # print(t1)
    # print(torch.transpose(t1, 0, 1))
    t1 = torch.unsqueeze(t1, 2)
    print(torch.sort(t1))

    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.memory_allocated())
