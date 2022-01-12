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
        self.psite = nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.pphen(x), self.env(x), self.psite(x)


def right_bin_class(output, original):
    return sum([1 for i in range(len(output)) if np.argmax(output[i]) == original[i]])


class nceloss:
    def __init__(self, weight):
        self.lossf = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))

    def loss(self, inpt, lbl):
        return self.lossf(inpt, lbl.long())

    def rightperc(self, output, original):
        output = output.detach().numpy()
        outdict = {str(i): 0 for i in original}
        for i in range(len(output)):
            if np.argmax(output[i]) == original[i]:
                outdict[str(original[i])] += 1
        return sum([1 for i in range(len(output)) if np.argmax(output[i]) == original[i]]), outdict

    def right_idx(self, opt, original):
        oux = opt.detach().numpy()
        return [i for i in range(len(opt)) if np.argmax(oux[i]) == original[i]]


class mcloss:
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
    return [item for sublist in t for item in sublist]


def generate_salency_map_nonabs(net, input, output, indexes, idx, losses, batchsize, strname,names,gennames):
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
        all_pos = []
        all_neg = []
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
            pos_vals = torch.clone(saliency).reshape(80 * 80).numpy()
            all_vals.append(pos_vals)

        all_vals = np.mean(np.array(all_vals), axis=0)
        normed_pos=all_vals.copy()
        normed_neg= -1*all_vals.copy()
        normed_pos[normed_pos<0]=0
        normed_neg[normed_neg<0]=0
        normed_pos = normed_pos.reshape(80, 80)
        normed_neg = normed_neg.reshape(80, 80)
        with open(f"./maps_pickle/{strname}_salency_map_{idx}_{out_idx}_{len(todoidxes)}.pickle", "wb")  as f:
            pickle.dump({"pos": normed_pos, "neg": normed_neg, "amt": len(todoidxes), "inidx": idx, "outidx": out_idx},
                        f)
        with open(f"maps_text/{strname}_{idx}_{out_idx}_{len(todoidxes)}_{names[out_idx]}_pos.txt","w") as f:
            f.write(gen_txt_importance({"pos": normed_pos, "neg": normed_neg},gennames,"pos"))
        with open(f"maps_text/{strname}_{idx}_{out_idx}_{len(todoidxes)}_{names[out_idx]}_neg.txt","w") as f:
            f.write(gen_txt_importance({"pos": normed_pos, "neg": normed_neg},gennames,"neg"))
        maxcorr = max(normed_pos.max(), normed_neg.max())
        normed_pos *= (255.0 / maxcorr)
        normed_neg *= (255.0 / maxcorr)
        rgbArray = np.zeros((80, 80, 3), 'uint8')
        rgbArray[..., 0] = normed_neg  # *256
        rgbArray[..., 2] = normed_pos  # *256
        im = Image.fromarray(rgbArray)
        im.save(f"maps/{strname}_salency_map_{idx}_{out_idx}_{len(todoidxes)}_{names[out_idx]}.png", format="png")


def genweight_classes(data):
    values, counts = np.unique(data, return_counts=True)
    values = values[1:]  # remove -1s
    counts = counts[1:]
    if max(data) + 1 != len(values):
        print("something is wrong", max(data), len(values), values)
        exit(-1)
    minval = min(counts)
    return [minval / j for j in counts]


def genweight_multi_classes(data):
    nws = np.array([sum(data[:, i]) for i in range(len(data[0]))])
    return np.array([min(nws) / sum(data[:, i]) for i in range(len(data[0]))])


def make_img(prefix, trainacc, testacc, loss, labels, classes_acc, classes_acc_test,namelist):
    fig, axs = plt.subplots(3 + len(classes_acc) * 2, figsize=(20, 15 * (3 + len(classes_acc) * 2)))
    fig.suptitle(prefix)
    x = list(range(len(trainacc[0])))
    for i in range(len(trainacc)):
        axs[0].plot(x, trainacc[i], label=labels[i], linewidth=4.0)
        axs[2].plot(x, loss[i], label=labels[i], linewidth=4.0)
    for i in range(len(labels)):
        axs[1].plot(x, testacc[i], label=labels[i], linewidth=4.0)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[0].set_title('Training acc')
    axs[1].set_title('Test acc')
    axs[2].set_title('Loss')
    for cacc in range(len(classes_acc)):
        x, y, lbls = make_class_img(classes_acc[cacc],namelist[cacc])
        for c in y:
            axs[3 + cacc].plot(x, c, linewidth=4.0)
        axs[3 + cacc].legend(lbls)
        axs[3 + cacc].set_ylim([0, 100])
        axs[3 + cacc].set_title(labels[cacc] + "_classes training")

    for cacc in range(len(classes_acc)):
        x, y, lbls = make_class_img(classes_acc_test[cacc],namelist[cacc])
        for c in y:
            axs[3 + len(classes_acc) + cacc].plot(x, c, linewidth=4.0)
        axs[3 + len(classes_acc) + cacc].legend(lbls)
        axs[3 + len(classes_acc) + cacc].set_ylim([0, 100])
        axs[3 + len(classes_acc) + cacc].set_title(labels[cacc] + "_classes test")
    axs[0].set_ylim([0, 100])
    axs[1].set_ylim([0, 100])

    plt.savefig(prefix + "_mainimg.png")
    plt.clf()
    plt.close(fig)


def make_class_img(dictlist,namelist):
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


net = Net()
'''

baselr=0.001
images = np.array([np.random.rand(1, 80, 80) for _ in range(200)])
output_classes_pphen = np.array([np.random.randint(2) for _ in range(200)])
output_classes_env = np.array([[np.random.randint(2) for _ in range(7)] for _ in range(100)])



loss = [nceloss(genweight_classes(output_classes_pphen)),mcloss(genweight_multi_classes(output_classes_env),0.9)]
optimizer = [optim.Adam(net.parameters(), lr=baselr),optim.Adam(net.parameters(), lr=baselr*(len(output_classes_pphen)/len(output_classes_env)))] # adapt learning rate to be equal for all outputs
bsize = 30

trainidx=list(range(100))

for _ in range(30):
    print("train:",train_epoch(net, loss, optimizer, images, output_classes_pphen, trainidx,bsize,0))
    print("train:",train_epoch(net, loss, optimizer, images, output_classes_env, trainidx,bsize,1))
    print("test:",checkdata(net,images,output_classes_pphen,trainidx,bsize,0,loss))
generate_salency_map(net,images,output_classes_pphen,trainidx,0,loss,32)
generate_salency_map(net,images,output_classes_env,trainidx,1,loss,32)
'''


def loadpick(file):
    with open("./data/pickle_files/" + file, "rb") as f:
        return pickle.load(f)


def gen_txt_importance(dta,genenames,key):
    dta=dta[key].reshape(-1)
    maxval=max(dta)
    ttval=sum(dta)
    rtx=['genename\trelative perc\ttotalperc\tgradient']
    for x in np.argsort(dta)[::-1]:
        if dta[x] != 0:
            rtx.append(f'{genenames[x]}\t{round(100*(dta[x]/maxval),2)}\t{round(100*(dta[x]/ttval),2)}\t{dta[x]}')
    return "\n".join(rtx)
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

env_training = loadpick("env_training.p")
env_test = loadpick("env_test.p")
env_train = [dkey_to_idx[i] for i in env_training]
env_test = [dkey_to_idx[i] for i in env_test]

pphen_training = loadpick("pphen_training.p")
pphen_test = loadpick("pphen_test.p")
pphen_train = [dkey_to_idx[i] for i in pphen_training]
pphen_test = [dkey_to_idx[i] for i in pphen_test]

site_training = loadpick("psite_training.p")
site_test = loadpick("psite_test.p")
site_train = [dkey_to_idx[i] for i in site_training]
site_test = [dkey_to_idx[i] for i in site_test]

with open("./data/pickle_files/gene_list.p","rb") as f:
    genenames=np.array(pickle.load(f))

base_data = len(pphen_training)
base_lr = 0.001

lr_pphen = base_lr * (base_data / len(pphen_training))
lr_env = base_lr * (base_data / len(env_training))
lr_psite = base_lr * (base_data / len(site_training))
print("lrs:", lr_pphen, lr_env, lr_psite)

print("weights:", genweight_classes(pphen_output), genweight_classes(site_output))

loss = [nceloss(genweight_classes(pphen_output)), mcloss(genweight_multi_classes(env_output), 0.9),
        nceloss(genweight_classes(site_output))]
optimizer = [optim.Adam(net.parameters(), lr=lr_pphen), optim.Adam(net.parameters(), lr=lr_env),
             optim.Adam(net.parameters(), lr=lr_psite)]  # adapt learning rate to be equal for all outputs
bsize = 32

prefix = "all_overnight"
t_to_idx = {"pphen": 0, "env": 1, "psite": 2}
train = [[] for i in t_to_idx.values() if i != -1]
test = [[] for i in t_to_idx.values() if i != -1]
losses = [[] for i in t_to_idx.values() if i != -1]
class_accs = [[] for i in t_to_idx.values() if i != -1]
class_accs_tst = [[] for i in t_to_idx.values() if i != -1]
labels = [i for i in t_to_idx if t_to_idx[i] != -1]
names=[None,None,None]
if t_to_idx["pphen"] != -1:
    names[t_to_idx["pphen"]]=pphen_out_names
if t_to_idx["env"] != -1:
    names[t_to_idx["env"]]=env_out_names
if t_to_idx["psite"] != -1:
    names[t_to_idx["psite"]]=psite_out_names
for i in range(300):
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

        generate_salency_map_nonabs(net, input_data, pphen_output, pphen_test, 0, loss, bsize, f'{prefix}_ep_{i}_phen',pphen_out_names,genenames)

    if t_to_idx["env"] != -1:
        acc, cacc = checkdata(net, input_data, env_output, env_test, bsize, 1, loss)
        test[t_to_idx["env"]].append(acc)
        class_accs_tst[t_to_idx["env"]].append(cacc)
        generate_salency_map_nonabs(net, input_data, env_output, env_test, 1, loss, bsize, f'{prefix}_ep_{i}_env',env_out_names,genenames)

    if t_to_idx["psite"] != -1:
        acc, cacc = checkdata(net, input_data, site_output, site_test, bsize, 2, loss)
        test[t_to_idx["psite"]].append(acc)
        class_accs_tst[t_to_idx["psite"]].append(cacc)
        generate_salency_map_nonabs(net, input_data, site_output, site_test, 2, loss, bsize, f'{prefix}_ep_{i}_site',psite_out_names,genenames)

    make_img(prefix, train, test, losses, labels, class_accs, class_accs_tst,names)
    torch.save(net.state_dict(), f"./nets/{prefix}_net_{i}.sdict")
