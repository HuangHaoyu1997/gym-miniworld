import numpy as np
import pickle,random,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from net import locomotion_net, retrival_net
import torch.optim as optim

os.environ['CUDA_VISIBLE_DEVICES']='5,6,7'
print('free GPU ID:',torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def process_data(dir):
    img,pos = [],[]
    file_list = os.listdir(dir)
    for file in file_list:
        with open(dir+file,'rb') as f:
            data = pickle.load(f)

        for i in range(len(data)):
            img.append(data[i][0])
            pos.append(data[i][1])
    return np.array(img), np.array(pos)

def train(net,img,pos,device,batch_size=64,lr=1e-4,epoch=100):
    criterion = nn.MSELoss() # nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr, betas=(0.9,0.999),eps=1e-8)

    data_num = len(img)
    iter_num = int(data_num / batch_size)
    for epoch in range(epoch):  # loop over the dataset multiple times
        epoch_loss = 0.0
        sample_list_1 = random.sample(list(range(data_num)),data_num)
        sample_list_2 = random.sample(list(range(data_num)),data_num)
        for i in range(iter_num):
            # 随机采样两次，一次采样batch_size个img，组成img pair
            data_batch = np.array([img[sample_list_1[i*batch_size:(i+1)*batch_size]],  
                        img[sample_list_2[i*batch_size:(i+1)*batch_size]]])
            
            # 计算两个img间距
            x_delta = np.sqrt(np.sum((pos[sample_list_1[i*batch_size:(i+1)*batch_size]] \
                                    - pos[sample_list_2[i*batch_size:(i+1)*batch_size]])**2,axis=-1))
            
            # 距离足够近，label=1，否则=0
            label_batch = np.zeros_like(x_delta)
            label_batch[x_delta<=0.15*6] = 1

            inputs, labels = torch.tensor(data_batch,dtype=torch.float32,device=device)/255., \
                             torch.tensor(label_batch,dtype=torch.float32,device=device)
            optimizer.zero_grad()
            # print(inputs[0])

            outputs = net(inputs[0].mean(dim=-1),inputs[1].mean(dim=-1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.cpu().item()
            # if i % 100 == 99:    # print every 100 mini-batches
            #     print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, epoch_loss / 100))
            #     epoch_loss = 0.0
        print(epoch, epoch_loss/iter_num)

    print('Finished Training')

if __name__ == "__main__":
    dir = '/home/hhy/navi_py/data/'
    img, pos = process_data(dir)
    print(img.shape,pos.shape)
    r_net = retrival_net().to(device)
    # aa = torch.tensor(img[0:16],dtype=torch.float32).mean(dim=-1)/255.
    # bb = torch.tensor(img[16:32],dtype=torch.float32).mean(dim=-1)/255.

    # result = r_net(aa,bb)
    # print(result.shape)
    train(r_net,img,pos,device)