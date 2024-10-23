import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import cv2

def has_cuda():
    return torch.cuda.is_available()

def which_device():
    return torch.device("cuda" if has_cuda() else "cpu")

def init_seed(args):
    torch.manual_seed(args.seed)

    if has_cuda():
        torch.cuda.manual_seed(args.seed)

        if has_cuda():
            torch.cuda.manual_seed_all(args.seed)


def show_model_summary(model, input_size):
    print(summary(model, input_size=input_size))

def imshow(im, title):
    npim = im.numpy()
    fig = plt.figure(figsize=(15, 7))
    plt.imshow(np.transpose(npim, (1, 2, 0)))  
    plt.title(title)

