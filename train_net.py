import numpy as np
import pickle,random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from net import locomotion_net, retrival_net
import torch.optim as optim





def process_data(dir):
    with open(dir,'rb') as f:
        data = pickle.load(f)
    
    img,pos = [],[]
    for i in range(len(data)):
        img.append(data[i][0])
        pos.append(data[i][1])
    return np.array(img), np.array(pos)

def train(net,img,pos,batch_size=16,lr=1e-3):
    criterion = nn.MSELoss() # nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)

    data_num = len(img)
    iter_num = int(data_num / batch_size)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
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

            inputs, labels = torch.tensor(data_batch,dtype=torch.float32)/255., \
                             torch.tensor(label_batch,dtype=torch.float32)
            optimizer.zero_grad()
            # print(inputs[0])

            outputs = net(inputs[0].mean(dim=-1),inputs[1].mean(dim=-1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":
    dir = '1.pkl'
    img, pos = process_data(dir)
    
    r_net = retrival_net()
    aa = torch.tensor(img[0:16],dtype=torch.float32).mean(dim=-1)/255.
    bb = torch.tensor(img[16:32],dtype=torch.float32).mean(dim=-1)/255.

    result = r_net(aa,bb)
    # print(result.shape)
    train(r_net,img,pos,)