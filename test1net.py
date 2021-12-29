import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import random


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 8, 3)
        self.conv4 = nn.Conv2d(8, 8, 3)
        self.fc1 = nn.Linear(72, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def right_bin_class(output, original):
    return sum([1 for i in range(len(output)) if np.argmax(output[i]) == original[i]])

def train_epoch(net, loss, optimizer, input, output, batches):
    random.shuffle(batches)
    rightamt = 0
    lossavg = 0
    net.train()
    for batch in batches:
        random.shuffle(batch)
        optimizer.zero_grad()
        x_np = torch.from_numpy(input[batch]).float()
        out = net(x_np)
        lls = loss(out, torch.from_numpy(output[batch]).long())
        lls.backward()
        optimizer.step()
        realout = torch.softmax(out, 1).detach().numpy()
        rightamt += right_bin_class(realout, output[batch])
        lossavg += lls.sum().detach().numpy()
    return rightamt, lossavg

def checkdata(net,input,output,batches):
    net.eval()
    rightamt = 0
    with torch.no_grad():
        for batch in batches:
            x_np = torch.from_numpy(input[batch]).float()
            out = net(x_np)
            realout = torch.softmax(out, 1).detach().numpy()
            rightamt += right_bin_class(realout, output[batch])
    return rightamt
net = Net()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
images = np.array([np.random.rand(3, 80, 80) for _ in range(200)])
output_classes = np.array([np.random.randint(2) for _ in range(200)])
bsize = 30
print(images[0:3].shape)
for _ in range(300):
    print(train_epoch(net, loss, optimizer, images, output_classes, [[1, 3, 5, 7], [2, 6, 3, 7]]))
    print("test:",checkdata(net,images,output_classes,[[9,4,5,2],[16,17,18,19]]))
