import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# print(new_model(torch.randn(1,3,224,224)).shape)
# print(res18(torch.randn(1,3,224,224)).shape)

class retrival_net(nn.Module):
    def __init__(self, input_channel=24):
        super(retrival_net,self).__init__()
        self.net = torchvision.models.resnet18(input_channel=input_channel)
        self.net1 = torch.nn.Sequential(*(list(self.net.children())[:-1])) # 去掉最后一层mlp
        self.net2 = torch.nn.Sequential(*(list(self.net.children())[:-1])) # 去掉最后一层mlp

        self.fc = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.net1(x1).squeeze(-1).squeeze(-1)
        x2 = self.net2(x2).squeeze(-1).squeeze(-1)
        x_cat = torch.cat((x1,x2),-1)
        out = self.fc(x_cat)
        return out

class locomotion_net(nn.Module):
    def __init__(self,action_dim=4, input_channel=24):
        super(locomotion_net,self).__init__()
        self.action_dim = action_dim
        self.net = torch.nn.Sequential(*(list(torchvision.models.resnet18(input_channel=input_channel).children())[:-1]))
        self.fc = nn.Linear(512, self.action_dim)
    
    def forward(self,x1,x2):
        x = torch.cat((x1,x2),-1)
        x = self.net(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        x = F.softmax(x,dim=-1)
        return x

if __name__ == "__main__":
    net = locomotion_net()
    r = net(torch.randn(1,3,100,200),torch.randn(1,3,100,200))
    print(r)