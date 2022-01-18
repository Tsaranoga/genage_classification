import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import accuracy_score
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2
import os
class Net(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 8, 3, padding=1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.pphen = nn.Linear(64, 2)
        self.env = nn.Linear(64, 7)
        self.psite = nn.Linear(64, 5)
        self.drop=nn.Dropout(p=dropout/100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x=self.drop(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.pphen(x), self.env(x), self.psite(x)
class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 8, 5)
        self.conv4 = nn.Conv2d(8, 8, 5)
        self.fc1 = nn.Linear(1800, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.pphen = nn.Linear(256, 2)
        self.env = nn.Linear(256, 7)
        self.psite = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        #
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.pphen(x), self.env(x), self.psite(x)
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 5, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 5, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 8, 5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(72, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(p=0.2),
        )
        self.pphen = nn.Sequential(nn.Linear(256, 128),
                                   nn.Tanh(), nn.Linear(128, 2))
        self.env = nn.Sequential(nn.Linear(256, 128),
                                 nn.Tanh(), nn.Linear(128, 7))
        self.psite = nn.Sequential(nn.Linear(256, 128),
                                   nn.Tanh(), nn.Linear(128, 5))

    def forward(self, x):
        x = self.seq(x)
        return self.pphen(x), self.env(x), self.psite(x)


class nceloss:
    def __init__(self, weight):
        '''
        creates a normal cross entropy loss object with the right_idx and rightperc functions to it that the network can work with different losses
        :param weight:
        '''
        self.lossf = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))

    def loss(self, inpt, lbl):
        '''
        returns the loss of the network for that batch
        :param inpt: predicted labels
        :param lbl: true labels
        :return:
        '''
        return self.lossf(inpt, lbl.long())

    def rightperc(self, output, original):
        '''
        calculates how many predictions were right and which class got right how much
        :param output:
        :param original:
        :return:
        '''
        output = output.detach().numpy()
        outdict = {str(i): 0 for i in original}
        for i in range(len(output)):
            if np.argmax(output[i]) == original[i]:
                outdict[str(original[i])] += 1
        return sum([1 for i in range(len(output)) if np.argmax(output[i]) == original[i]]), outdict

    def right_idx(self, opt, original):
        '''
        gives back the right predicted indexes
        :param opt:
        :param original:
        :return:
        '''
        oux = opt.detach().numpy()
        return [i for i in range(len(opt)) if np.argmax(oux[i]) == original[i]]


class mcloss:
    '''
    all line in nceloss but for  multiclass labels, also a threshhold for when we think something is predicted right
    '''
    def __init__(self, weight, threshhold):
        print(weight)
        self.lossf = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(weight))
        self.thresh = threshhold

    def loss(self, inpt, lbl):
        return self.lossf(inpt, lbl.float())

    def rightperc(self, output, original):
        oux = torch.sigmoid(output.detach()).numpy()
        outdict = {str(i): 0 for i in original}
        for i in range(len(output)):
            outdict[str(original[i])] += accuracy_score(original[i], oux[i].round())
        return sum([accuracy_score(original[i], oux[i].round()) for i in range(len(output))]), outdict

    def right_idx(self, opt, original):
        oux = torch.sigmoid(opt.detach()).numpy()
        return [i for i in range(len(oux)) if accuracy_score(original[i], oux[i].round()) > self.thresh]


def train_epoch(net, losses, optimizers, input, output, indexes, batchsize, idx):
    '''
    trains the network with the given indexes and uses the idx to train the right output
    :param net: pytorch network with multiple outputs
    :param losses: list of loss class objects (mcloss or nceloss)
    :param optimizers: list of optimizers
    :param input: all the imput data, not all will be used
    :param output: the output data, some will have default values, when they are either used in test/eval or not known
    :param indexes: the indexes of input to output that will be trained
    :param batchsize: how many will be trained in a batch together
    :param idx: the index of the output/losses/optimizers
    :return: percent accuracy, average loss, accuracy for each class
    '''
    rightamt = 0
    lossavg = 0
    net.train()
    classesright = {str(i): 0 for i in output}
    for batch in batches(indexes, batchsize):
        net.zero_grad()
        x_np = torch.from_numpy(input[batch]).float()
        out = net(x_np)[idx]
        lls = losses[idx].loss(out, torch.from_numpy(output[batch]))
        lls.backward()
        optimizers[idx].step()
        rgt, class_acc = losses[idx].rightperc(out, output[batch])
        rightamt += rgt
        for i in class_acc:
            classesright[i] += class_acc[i]
        lossavg += lls.sum().detach().numpy()
    ttx = [str(i) for i in output[indexes]]
    values, counts = np.unique(ttx, return_counts=True)
    for i in range(len(values)):
        classesright[values[i]] = 100 * classesright[values[i]] / counts[i]

    return 100 * rightamt / len(indexes), lossavg / len(indexes), classesright


def batches(l, n):
    '''
    splits the indexes in random batches
    :param l: indexes
    :param n: batchsize
    :return: list of batches
    '''
    random.shuffle(l)
    return [l[i:i + n] for i in range(0, len(l), n)]


def checkdata(net, input, output, indexes, batchsize, idx, losses):
    net.eval()
    rightamt = 0
    classesright = {str(i): 0 for i in output}
    with torch.no_grad():
        for batch in batches(indexes, batchsize):
            x_np = torch.from_numpy(input[batch]).float()
            out = net(x_np)[idx]
            rgt, class_acc = losses[idx].rightperc(out, output[batch])
            rightamt += rgt
            for i in class_acc:
                classesright[i] += class_acc[i]
    ttx = [str(i) for i in output[indexes]]
    values, counts = np.unique(ttx, return_counts=True)
    for i in range(len(values)):
        classesright[values[i]] = 100 * classesright[values[i]] / counts[i]
    return 100 * rightamt / len(indexes), classesright


def flatten(t):
    '''
    just flatten the list
    :param t:
    :return:
    '''
    return [item for sublist in t for item in sublist]


def generate_salency_map_nonabs(net, input, output, indexes, idx, losses, batchsize, strname,names,gennames,epnr):
    '''
    generates the salency maps as well as the text files for the network
    :param net:
    :param input:
    :param output:
    :param indexes:
    :param idx:
    :param losses:
    :param batchsize:
    :param strname:
    :param names:
    :param gennames:
    :return:
    '''
    net.eval()
    right_idxes = []
    with torch.no_grad():
        for batch in batches(indexes, batchsize):
            x_np = torch.from_numpy(input[batch]).float()
            out = net(x_np)[idx]
            right_idxes.append([batch[right] for right in losses[idx].right_idx(out, output[batch])])
    right_idxes = flatten(right_idxes)
    if isinstance(output[0], np.int64):
        out = np.unique(output)
    else:
        out = []
        for i in range(len(output[0])):
            if sum([output[j][i] for j in range(len(output))]) > 0:
                out.append(i)
    for out_idx in out:
        if isinstance(output[0], np.int64):
            todoidxes = [i for i in right_idxes if output[i] == out_idx]
        else:
            todoidxes = [i for i in right_idxes if output[i][out_idx] == 1]
        all_vals=[]
        if len(todoidxes) == 0:
            # print("not doing",out_idx, "from",idx)
            continue
        for index in todoidxes:
            inpt = torch.from_numpy(input[index]).float()[None, :, :, :]
            inpt.requires_grad_()
            net_out = net(inpt)[idx]
            net_out[0, out_idx].backward()
            saliency = inpt.grad.data
            pos_vals = torch.clone(saliency).reshape(image_size * image_size).numpy()
            all_vals.append(pos_vals)

        all_vals = np.mean(np.array(all_vals), axis=0)
        normed_pos=all_vals.copy()
        normed_neg= -1*all_vals.copy()
        normed_pos[normed_pos<0]=0
        normed_neg[normed_neg<0]=0
        normed_pos = normed_pos.reshape(image_size, image_size)
        normed_neg = normed_neg.reshape(image_size, image_size)
        maxcorr = max(normed_pos.max(), normed_neg.max())
        if maxcorr==0:
            continue
        with open(f"./graphs/{strname}/maps_pickle/{epnr}salency_map_{idx}_{out_idx}_{len(todoidxes)}.pickle", "wb")  as f:
            pickle.dump({"pos": normed_pos, "neg": normed_neg, "amt": len(todoidxes), "inidx": idx, "outidx": out_idx},
                        f)
        with open(f"./graphs/{strname}/maps_text/{epnr}_{idx}_{out_idx}_{len(todoidxes)}_{names[out_idx]}_pos.txt","w") as f:
            f.write(gen_txt_importance({"pos": normed_pos, "neg": normed_neg},gennames,"pos"))
        with open(f"./graphs/{strname}/maps_text/{epnr}_{idx}_{out_idx}_{len(todoidxes)}_{names[out_idx]}_neg.txt","w") as f:
            f.write(gen_txt_importance({"pos": normed_pos, "neg": normed_neg},gennames,"neg"))
        with open(f"./graphs/{strname}/maps_text/{epnr}_{idx}_{out_idx}_{len(todoidxes)}_{names[out_idx]}_zero.txt","w") as f:
            f.write(gen_txt__not_importance({"pos": normed_pos, "neg": normed_neg},gennames))

        normed_pos *= (255.0 / maxcorr)
        normed_neg *= (255.0 / maxcorr)
        rgbArray = np.zeros((image_size, image_size, 3), 'uint8')
        rgbArray[..., 0] = normed_neg  # *256
        rgbArray[..., 2] = normed_pos  # *256
        im = Image.fromarray(rgbArray)
        im.save(f"./graphs/{strname}/maps/{epnr}_salency_map_{idx}_{out_idx}_{len(todoidxes)}_{names[out_idx]}.png", format="png")


def genweight_classes(data):
    '''
    generates weight losses for normal cross entropy labels
    :param data:
    :return:
    '''
    values, counts = np.unique(data, return_counts=True)
    values = values[1:]  # remove -1s
    counts = counts[1:]
    if max(data) + 1 != len(values):
        print("something is wrong", max(data), len(values), values)
        exit(-1)
    minval = max(counts)
    return [minval / j for j in counts]


def genweight_multi_classes(data):
    '''
    generates weights for multiclass loss
    :param data:
    :return:
    '''
    nws = np.array([sum(data[:, i]) for i in range(len(data[0]))])
    return np.array([max(nws) / sum(data[:, i]) for i in range(len(data[0]))])


def make_img(prefix, trainacc, testacc, loss, labels, classes_acc, classes_acc_test,namelist):
    '''
    makes/overwrites the current training image
    :param prefix:
    :param trainacc:
    :param testacc:
    :param loss:
    :param labels:
    :param classes_acc:
    :param classes_acc_test:
    :param namelist:
    :return:
    '''
    if not os.path.isdir("./graphs/"+prefix):
        os.mkdir("./graphs/"+prefix)

    x = list(range(len(trainacc[0])))
    for i in range(len(trainacc)):
        plt.plot(x, trainacc[i], label=labels[i], linewidth=4.0)
    plt.ylim((0, 100))
    plt.legend()
    plt.title("Training accuracy")
    plt.xlabel('Epoch')
    plt.ylabel("Accuracy")
    plt.savefig("./graphs/"+prefix+"/training.png")
    plt.clf()

    for i in range(len(trainacc)):
        plt.plot(x, loss[i], label=labels[i], linewidth=4.0)
    plt.legend()
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel("Error")
    plt.savefig("./graphs/"+prefix+"/loss.png")
    plt.clf()

    for i in range(len(trainacc)):
        plt.plot(x, testacc[i], label=labels[i], linewidth=4.0)
    plt.ylim((0, 100))
    plt.legend()
    plt.title("Test accuracy")
    plt.xlabel('Epoch')
    plt.ylabel("Accuracy")
    plt.savefig("./graphs/"+prefix+"/test.png")
    plt.clf()

    for cacc in range(len(classes_acc)):
        x, y, lbls = make_class_img(classes_acc[cacc],namelist[cacc])
        for c in y:
            plt.plot(x, c, linewidth=4.0)
        plt.legend(lbls)
        plt.ylim((0, 100))
        plt.title(labels[cacc] + " classes training")
        plt.xlabel('Epoch')
        plt.ylabel("Accuracy")
        plt.savefig("./graphs/"+prefix+"/class"+str(cacc)+"_training.png")
        plt.clf()


    for cacc in range(len(classes_acc)):
        x, y, lbls = make_class_img(classes_acc_test[cacc],namelist[cacc])
        for c in y:
            plt.plot(x, c, linewidth=4.0)
        plt.legend(lbls)
        plt.ylim((0, 100))
        plt.title(labels[cacc] + " classes test")
        plt.xlabel('Epoch')
        plt.ylabel("Accuracy")
        plt.savefig("./graphs/"+prefix+"/class"+str(cacc)+"_test.png")
        plt.clf()

    with open("./graphs/"+prefix+"/accsach.txt","w") as f:
        for i in range(len(trainacc)):
            trainbest=np.argmax(trainacc[i])
            testbest= np.argmax(testacc[i])
            f.write(f'trainig: {trainbest}:{trainacc[i][trainbest]}   test: {testbest} : {testacc[i][testbest]}\n' )


def make_class_img(dictlist,namelist):
    '''
    outsourced from make_img , generates the graphs for the specific classes with real labels
    :param dictlist:
    :param namelist:
    :return:
    '''
    dictlist[0].pop('-1', None)
    dictlist[0].pop('[0 0 0 0 0 0 0]', None)
    labels = list(dictlist[0].keys())
    reallabels=[]
    if labels[0][0] !="[":
        reallabels=[namelist[int(i)] for i in range(len(labels))]
    else:
        for lbl in labels:
            #print(labels)
            lblary=eval(lbl.replace(" ",","))
            name=""
            for i in range(len(lblary)):
                if lblary[i]!=0:
                    name +=namelist[i] + " "
            reallabels.append(name)
    x = list(range(len(dictlist)))
    y = []
    for l in labels:
        y.append([dictlist[i][l] for i in range(len(dictlist))])
    return x, y, reallabels




def loadpick(file):
    with open("./data/pickle_files/" + file, "rb") as f:
        return pickle.load(f)


def gen_txt_importance(dta,genenames,key):

    dta1=dta["pos"].reshape(-1)
    dta2=dta["neg"].reshape(-1)
    dta=dta[key].reshape(-1)
    maxval=max(dta)
    ttval=sum(dta)
    minthresh=0
    rtx=['genename\trelative perc\ttotalperc\tgradient']
    for i in range(len(genenames)):
        if genenames[i]=="-":
            minthresh=max(minthresh,dta1[i],dta2[i])
    for x in np.argsort(dta)[::-1]:
        if dta[x] > minthresh:
            rtx.append(f'{genenames[x]}\t{round(100*(dta[x]/maxval),2)}\t{round(100*(dta[x]/ttval),2)}\t{dta[x]}')
    return "\n".join(rtx)



def gen_txt__not_importance(dta,genenames):
    dta1=dta["pos"].reshape(-1)
    dta2=dta["neg"].reshape(-1)
    rtx=['genename']
    minthresh=0
    # set the minthresh to the highest value were we added zeros
    for i in range(len(genenames)):
        if genenames[i]=="-":
            minthresh=max(minthresh,dta1[i],dta2[i])
    for x in range(len(dta1)):
        if dta1[x] <= minthresh and dta2[x] <= minthresh:
            rtx.append(f'{genenames[x]}')
    return "\n".join(rtx)


def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0)] * array.ndim

    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def gen_inputmap(input_data,indexes,filename):
    phen_maps=input_data[indexes]
    meanstuff=np.max(phen_maps,axis=0).reshape(image_size,image_size)
    #meanstuff -=1
    #print(sum(meanstuff))
    #return
    cv2.imwrite(f'{filename}.png', meanstuff*255)

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


def test_mean_classify(input_data,trainidx,testidx,output):
    classmaps=[np.mean(input_data[[i for i in trainidx if output[i]==0]],axis=0).reshape(image_size*image_size),
               np.mean(input_data[[i for i in trainidx if output[i]==1]],axis=0).reshape(image_size*image_size),
               ]
    right=0
    for t_idx in testidx:
        ipx=input_data[t_idx].reshape(image_size*image_size)
        c1=sum(abs(ipx-classmaps[0]))
        c2=sum(abs(ipx-classmaps[1]))
        if c1 <c2 and output[t_idx]==0  or c1 >c2 and output[t_idx]==1:
            right+=1
    print(right,right/len(testidx))

dropout=30
image_size=72
net = Net(dropout)
inputstuff = loadpick("input.p")
input_keys = list(inputstuff.keys())
dkey_to_idx = {input_keys[i]: i for i in range(len(input_keys))}

input_data = np.expand_dims(np.array([inputstuff[i] for i in input_keys]), axis=1)

gt_site = loadpick("gt_pa_site.p")
psite_out_names=list(gt_site["2693429607"].keys())

inputstuff = loadpick("input.p")
input_keys = list(inputstuff.keys())
dkey_to_idx = {input_keys[i]: i for i in range(len(input_keys))}

input_data = np.expand_dims(np.array([inputstuff[i] for i in input_keys]), axis=1)

gt_site = loadpick("gt_pa_site.p")
psite_out_names=list(gt_site["2693429607"].keys())


site_output = [-1 for _ in range(len(input_data))]
for k in gt_site:
    site_output[dkey_to_idx[k]] = np.argmax(tuple(gt_site[k].values()))
    if sum(gt_site[k].values()) == 0:
        site_output[dkey_to_idx[k]] = 3
site_output = np.array(site_output)

unique,counts =np.unique(site_output,return_counts=True)
print("psite counts:",unique,counts)

gt_pphen = loadpick("gt_pa_pheno.p")

pphen_out_names=list(gt_pphen["2786546145"].keys())
pphen_output = [-1 for _ in range(len(input_data))]

for k in gt_pphen:
    pphen_output[dkey_to_idx[k]] = np.argmax(list(gt_pphen[k].values()))
pphen_output = np.array(pphen_output)

gt_env = loadpick("gt_environment.p")
env_out_names=list(gt_env["2786546145"].keys())
env_output = [[0, 0, 0, 0, 0, 0, 0] for _ in range(len(input_data))]
for k in gt_env:
    env_output[dkey_to_idx[k]] = list(gt_env[k].values())
env_output = np.array(env_output)

unique,counts =np.unique([str(i) for i in env_output],return_counts=True)
print(unique,counts,env_output)

env_training = loadpick("env_training.p")
env_test = loadpick("env_test.p")
env_train = [dkey_to_idx[i] for i in env_training]
env_test = [dkey_to_idx[i] for i in env_test]

pphen_training = loadpick("pphen_training.p")
pphen_test = loadpick("pphen_test.p")
pphen_train = [dkey_to_idx[i] for i in pphen_training]
pphen_test = [dkey_to_idx[i] for i in pphen_test]
pphen_val = loadpick("pphen_validation.p")
pphen_val = [dkey_to_idx[i] for i in pphen_val]

site_training = loadpick("psite_training.p")
site_test = loadpick("psite_test.p")
site_train = [dkey_to_idx[i] for i in site_training]
site_test = [dkey_to_idx[i] for i in site_test]

with open("./data/pickle_files/gene_list.p","rb") as f:
    genenames=pickle.load(f)
while len(genenames)!= image_size*image_size:
    genenames.append("-")
genenames=np.array(genenames)
base_data = len(pphen_training)
base_lr = 0.001

lr_pphen = base_lr * (base_data / len(pphen_training))
lr_env = base_lr * (base_data / len(env_training))
lr_psite = base_lr * (base_data / len(site_training))
print("lrs:", lr_pphen, lr_env, lr_psite)

print("weights:", genweight_classes(pphen_output), genweight_classes(site_output),genweight_multi_classes(env_output))
#exit(0)
loss = [nceloss(genweight_classes(pphen_output)), mcloss(genweight_multi_classes(env_output), 0.9),
        nceloss(genweight_classes(site_output))]
optimizer = [optim.Adam(net.parameters(), lr=lr_pphen), optim.Adam(net.parameters(), lr=lr_env),
             optim.Adam(net.parameters(), lr=lr_psite)]  # adapt learning rate to be equal for all outputs
bsize = 32


test_mean_classify(input_data,pphen_train,pphen_test,pphen_output)


print(len(input_data))
print(input_data[0].shape)

prefix = f'batchsize {bsize} dropout {dropout}'
t_to_idx = {"pphen": 0, "env": -1, "psite": -1}
train = [[] for i in t_to_idx.values() if i != -1]
test = [[] for i in t_to_idx.values() if i != -1]
losses = [[] for i in t_to_idx.values() if i != -1]
class_accs = [[] for i in t_to_idx.values() if i != -1]
class_accs_tst = [[] for i in t_to_idx.values() if i != -1]
labels = [i for i in t_to_idx if t_to_idx[i] != -1]
names=[None,None,None]

# input changing  (abundance or zscore)
zscorenormalize=False
occurence_only=True
pphen_Genes_only=False
if pphen_Genes_only:
    prefix += " pphen genes only"
    datawewant=input_data[pphen_train + pphen_test + pphen_val]
    idx_to_Delete=np.max(datawewant,axis=0).reshape(image_size*image_size)
    idxwewanttodelete=[i for i in range(len(idx_to_Delete)) if idx_to_Delete[i] == 0]
    genenames = list(np.delete(genenames,idxwewanttodelete))
    input_data = np.delete(input_data.reshape(-1,image_size*image_size),idxwewanttodelete,axis=1)
    sizewewant=np.ceil(np.sqrt(input_data.shape[1]))
    image_size=int(sizewewant)
    while len(genenames)!= image_size*image_size:
        genenames.append("-")
    input_data =pad_along_axis(input_data,int(sizewewant**2),1).reshape(-1,1,image_size,image_size)

if occurence_only:
    prefix += " occurence"
    input_data = np.where(input_data == 0, 0, 1)

if zscorenormalize:
    prefix += " zscore normalized"
    normvals=[]
    ipf=input_data.reshape(-1,image_size*image_size)
    for gene in range(image_size*image_size):
        normdata=[i[gene] for i in ipf[pphen_train]]
        meanv=np.mean(normdata)
        stdv=np.std(normdata)
        if stdv==0:
            stdv=1
        normvals.append({"mean":meanv,"std":stdv})
        for i in range(ipf.shape[0]):
            ipf[i][gene]= (ipf[i][gene] - meanv)/stdv

    ipf=ipf.reshape(-1,1,image_size,image_size)
    input_data=ipf

if t_to_idx["pphen"] != -1:
    names[t_to_idx["pphen"]]=pphen_out_names
if t_to_idx["env"] != -1:
    names[t_to_idx["env"]]=env_out_names
if t_to_idx["psite"] != -1:
    names[t_to_idx["psite"]]=psite_out_names

gen_inputmap(input_data,[i for i in pphen_train if pphen_output[i]==0],f'pphen_{pphen_out_names[0]}')
gen_inputmap(input_data,[i for i in pphen_train if pphen_output[i]==1],f'pphen_{pphen_out_names[1]}')

gen_inputmap(input_data,[i for i in range(len(input_data))],f'all_data')

gen_inputmap(input_data,[i for i in pphen_train + pphen_test + pphen_val],f'all_data_pphen')

print(input_data.shape)

os.mkdir("./graphs/" + prefix)
os.mkdir("./graphs/" + prefix + "/maps")
os.mkdir("./graphs/" + prefix + "/maps_pickle")
os.mkdir("./graphs/" + prefix + "/maps_text")
os.mkdir("./graphs/" + prefix + "/nets")
for i in range(50):
    print(i)
    # train
    if t_to_idx["pphen"] != -1:
        acc, lss, cacc = train_epoch(net, loss, optimizer, input_data, pphen_output, pphen_train, bsize, 0)
        print("train pphen:", acc)
        train[t_to_idx["pphen"]].append(acc)
        losses[t_to_idx["pphen"]].append(lss)
        class_accs[t_to_idx["pphen"]].append(cacc)
    if t_to_idx["env"] != -1:
        acc, lss, cacc = train_epoch(net, loss, optimizer, input_data, env_output, env_train, bsize, 1)
        train[t_to_idx["env"]].append(acc)
        losses[t_to_idx["env"]].append(lss)
        class_accs[t_to_idx["env"]].append(cacc)
        print("train env:", acc)
    if t_to_idx["psite"] != -1:
        acc, lss, cacc = train_epoch(net, loss, optimizer, input_data, site_output, site_train, bsize, 2)
        train[t_to_idx["psite"]].append(acc)
        losses[t_to_idx["psite"]].append(lss)
        class_accs[t_to_idx["psite"]].append(cacc)
        print("train psite:", acc)

    # test

    if t_to_idx["pphen"] != -1:
        acc, cacc = checkdata(net, input_data, pphen_output, pphen_test, bsize, 0, loss)
        test[t_to_idx["pphen"]].append(acc)
        class_accs_tst[t_to_idx["pphen"]].append(cacc)
        print("test pphen:", acc)

        generate_salency_map_nonabs(net, input_data, pphen_output, pphen_test, 0, loss, bsize, prefix,pphen_out_names,genenames,i)

    if t_to_idx["env"] != -1:
        acc, cacc = checkdata(net, input_data, env_output, env_test, bsize, 1, loss)
        test[t_to_idx["env"]].append(acc)
        class_accs_tst[t_to_idx["env"]].append(cacc)
        generate_salency_map_nonabs(net, input_data, env_output, env_test, 1, loss, bsize, prefix,env_out_names,genenames,i)

    if t_to_idx["psite"] != -1:
        acc, cacc = checkdata(net, input_data, site_output, site_test, bsize, 2, loss)
        test[t_to_idx["psite"]].append(acc)
        class_accs_tst[t_to_idx["psite"]].append(cacc)
        generate_salency_map_nonabs(net, input_data, site_output, site_test, 2, loss, bsize,prefix,psite_out_names,genenames,i)

    make_img(prefix, train, test, losses, labels, class_accs, class_accs_tst,names)
    torch.save(net.state_dict(), f"./graphs/{prefix}/nets/{i}.sdict")
