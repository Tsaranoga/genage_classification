import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import accuracy_score
from PIL import Image
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 8, 3)
        self.conv4 = nn.Conv2d(8, 8, 3)
        self.fc1 = nn.Linear(72, 64)
        self.fc2 = nn.Linear(64, 64)
        self.pphen = nn.Linear(64, 2)
        self.env = nn.Linear(64, 7)
        self.psite = nn.Linear(64,4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.pphen(x),self.env(x),self.psite(x)


def right_bin_class(output, original):
    return sum([1 for i in range(len(output)) if np.argmax(output[i]) == original[i]])

class nceloss:
    def __init__(self):
        self.lossf= nn.CrossEntropyLoss()
    def loss(self,inpt,lbl):
        return self.lossf(inpt,lbl.long())
    def rightperc(self,output,original):
        return sum([1 for i in range(len(output)) if np.argmax(output[i]) == original[i]])
    def right_idx(self,opt,original):
        return [i for i in range(len(opt)) if np.argmax(opt[i]) == original[i]]

class mcloss:
    def __init__(self,threshhold):
        self.lossf= nn.BCEWithLogitsLoss()
        self.thresh=threshhold
    def loss(self,inpt,lbl):
        return self.lossf(inpt,lbl)
    def rightperc(self,output,original):
        output=torch.sigmoid(output.detach()).numpy()
        return sum([accuracy_score(original[i],output[i]) for i in range(len(output))])
    def right_idx(self,opt,original):
        oux=torch.sigmoid(opt.detach()).numpy()
        return [i for i in range(len(oux)) if accuracy_score(original[i],oux[i])>self.thresh  ]

def train_epoch(net, losses, optimizers, input, output, indexes,batchsize,idx):
    rightamt = 0
    lossavg = 0
    net.train()
    for batch in batches(indexes,batchsize):
        net.zero_grad()
        x_np = torch.from_numpy(input[batch]).float()
        out = net(x_np)[idx]
        lls = losses[idx].loss(out, torch.from_numpy(output[batch]))
        lls.backward()
        optimizers[idx].step()
        rightamt += losses[idx].rightperc(out.detach().numpy(), output[batch])
        lossavg += lls.sum().detach().numpy()
    return rightamt, lossavg


def batches(l, n):
    random.shuffle(l)
    return [l[i:i + n] for i in range(0, len(l), n)]

def checkdata(net,input,output,indexes,batchsize,idx,losses):
    net.eval()
    rightamt = 0
    with torch.no_grad():
        for batch in batches(indexes,batchsize):
            x_np = torch.from_numpy(input[batch]).float()
            out = net(x_np)[idx]
            rightamt += losses[idx].rightperc(out.detach().numpy(), output[batch])
    return rightamt
def flatten(t):
    return [item for sublist in t for item in sublist]
def generate_salency_map(net,input,output,indexes,idx,losses,batchsize):
    net.eval()
    right_idxes=[]
    print(output[0])

    with torch.no_grad():
        for batch in batches(indexes,batchsize):
            x_np = torch.from_numpy(input[batch]).float()
            out = net(x_np)[idx]
            right_idxes.append([batch[right] for right in losses[idx].right_idx(out.detach().numpy(), output[batch])])
    right_idxes= flatten(right_idxes)
    print(output[0])
    if isinstance(output[0], np.int32):
        out=np.unique(output)
    else:
        out=[]
        for i in range(len(output[0])):
            if sum([output[j][i] for j in range(len(output)) ])>0:
                out.append(i)

    print("ot:",out, isinstance(output[0], np.int32))
    for out_idx in out:
        if isinstance(output[0], np.int32):
            print("here")
            todoidxes=[i for i in right_idxes if output[i]==out_idx]
        else:
            print(output[0])
            todoidxes=[i for i in right_idxes if output[i][out_idx]==1]
        allsalencies=[]
        if len(todoidxes)==0:
            print("not doing",out_idx)
            continue
        print(out_idx,todoidxes)
        for index in todoidxes:
            inpt = torch.from_numpy(input[index]).float()[None,:,:,:]

            inpt.requires_grad_()
            net_out = net(inpt)[idx]
            net_out[0, out_idx].backward()
            saliency, _ = torch.max(inpt.grad.data.abs(), dim=1)

            saliency = saliency.reshape(80* 80).numpy()
            allsalencies.append(saliency)
        allsalencies=np.array(allsalencies)
        normed=np.mean(allsalencies,axis=0).reshape(80,80)
        print(normed.shape)
        normed *= (255.0/normed.max())
        im = Image.fromarray(normed)

        print(normed)
        im.convert("L").save("salency_map_"+str(idx) + "_"+str(out_idx) + "_" + str(len(todoidxes)) + ".png", format="png")


net = Net()
loss = [nceloss(),]
optimizer = [optim.Adam(net.parameters(), lr=0.001)]
images = np.array([np.random.rand(1, 80, 80) for _ in range(200)])
output_classes_pphen = np.array([np.random.randint(2) for _ in range(200)])
output_classes_pphen = np.array([np.random.randint(2) for _ in range(200)])
bsize = 30
print(images[0:3].shape)
trainidx=list(range(100))

for _ in range(30):
    print("train:",train_epoch(net, loss, optimizer, images, output_classes_pphen, trainidx,bsize,0))
    print("test:",checkdata(net,images,output_classes_pphen,trainidx,bsize,0,loss))
print(output_classes_pphen[0])
generate_salency_map(net,images,output_classes_pphen,trainidx,0,loss,32)
