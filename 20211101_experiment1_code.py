# Date: 2021-09-27 15:00:00
# from functools import lru_cache
from numpy.core.defchararray import array
from numpy.core.numeric import outer
from torch import nn
from torch.nn.modules import linear
from torch.utils.data import DataLoader, TensorDataset, dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch

# parsms
batch_size = 25
lr = 0.001
epoches = 10000
wid = 100
dep = 4
af = "ReLU"

# prepare datasets
x = np.linspace(start = 0, stop= 4*np.pi, num= 400, endpoint=False)
# do not contain 4*pi
y = np.sin(x)

# X = np.expand_dims(x,axis=1)
X = x.reshape(-1,1)
Y = y.reshape(400,-1)

# batch them with tensordataset
tX = torch.tensor(X,dtype=torch.float)
tY = torch.tensor(Y,dtype=torch.float)
datasets = TensorDataset(tX,tY)
dataloader = DataLoader(datasets,batch_size,shuffle=True)

# define the net
class cp_Net(torch.nn.Module):
    def __init__(self):
        super(cp_Net, self).__init__()
        self.net=torch.nn.Sequential(
            torch.nn.Linear(1,100),
            #torch.nn.Sigmoid(),
            torch.nn.ReLU(),
            #torch.nn.Tanh(),
            #torch.nn.LeakyReLU(1),
            #torch.nn.Softplus(),
            #torch.nn.ELU(0.1),
            torch.nn.Linear(100,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,1)
        )

    def forward(self, input:torch.FloatTensor):
        return self.net(input)
        
net=cp_Net()

# define optimizer and loss func
optim = torch.optim.Adam(cp_Net.parameters(net),lr)
Loss  = torch.nn.MSELoss()

# start training
lossres = []
for epoch in range(epoches):
    loss=None
    for batch_x,batch_y in dataloader:
        y_predict=net(batch_x)
        loss=Loss(y_predict,batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    # print loss every epoches/10 epochs
    if (epoch+1)%(epoches/10)==0:
        print("step: {0} , loss: {1}".format(epoch+1,loss.item()))
    if (epoch+1)%(epoches/100)==0:
        lossres += [loss.item()]

# plot the trained results and the real values.
def cp_plot(dep,wid,lr,af,epo,bs):
    plt.plot(x,y,label="actual")
    plt.plot(x,net(tX).detach().numpy(),label="predict")
    sf = "wid=" + str(wid) + " dep=" + str(dep) + " lr=" + str(lr) + " af=" + str(af) + " epo=" + str(epoches) + " bs=" + str(batch_size) 
    plt.title("sin(x) function " + sf)
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    plt.savefig(fname=sf + ".png")
    plt.show()
   

cp_plot(dep,wid,lr,af,epoches,batch_size)
plt.plot(np.linspace(1, int(epoches*400/batch_size), 100),np.array(lossres))
plt.xlabel("ite")
#plt.savefig(fname= str(int(epoches*400/batch_size))+ ".png")
#plt.savefig(fname= "lr="+ str(lr)+ ".png")
plt.savefig(fname = "wid=" + str(wid) + " dep=" + str(dep) + " epo=" + str(epoches)+ ".png")
plt.show()