import numpy as np
import pickle,random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from net import locomotion_net, retrival_net
import torch.optim as optim


batch_size = 16
dir = '1.pkl'

def process_data(dir):
    with open(dir,'rb') as f:
        data = pickle.load(f)
    
    img,pos = [],[]
    for i in range(len(data)):
        img.append(data[i][0])
        pos.append(data[i][1])
    return np.array(img), np.array(pos)

def train(net,train_data,lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
    
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in train_data:
            inputs, pos = data

            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":
    
    img, pos = process_data(dir)
    data_num = len(img)
    sample_list_1 = random.sample(list(range(data_num)),data_num)
    sample_list_2 = random.sample(list(range(data_num)),data_num)

    iter_num = int(data_num / batch_size)
    for i in range(iter_num):
        # 随机采样两次，一次采样batch_size个img，组成img pair
        data_batch = [img[sample_list_1[i*batch_size:(i+1)*batch_size]],  
                      img[sample_list_2[i*batch_size:(i+1)*batch_size]]]
        
        # 计算两个img间距
        x_delta = np.sqrt(np.sum((pos[sample_list_1[i*batch_size:(i+1)*batch_size]] \
                                - pos[sample_list_2[i*batch_size:(i+1)*batch_size]])**2,axis=-1))
        
        # 距离足够近，label=1，否则=0
        label_batch = 1 if x_delta <= 0.15*6 else 0
        # print(np.array(data_batch).shape, label_batch)

        img = 
    r_net = retrival_net()
    aa = torch.tensor(img[0:16],dtype=torch.float32).mean(dim=-1)/255.
    bb = torch.tensor(img[16:32],dtype=torch.float32).mean(dim=-1)/255.

    result = r_net(aa,bb)
    print(result.shape)