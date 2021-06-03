import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from net import locomotion_net, retrival_net



def process_data(dir):
    with open(dir,'rb') as f:
        data = pickle.load(f)
    
    img,pos = [],[]
    for i in range(len(data)):
        img.append(data[i][0])
        pos.append(data[i][1])
    return img,pos


if __name__ == "__main__":
    dir = '1.pkl'
    img, pos = process_data(dir)

    r_net = retrival_net()
    aa = torch.tensor(img[0:16],dtype=torch.float32).mean(dim=-1)/255.
    bb = torch.tensor(img[16:32],dtype=torch.float32).mean(dim=-1)/255.
    print(aa.shape)

    result = r_net(aa,bb)
    print(result.shape)