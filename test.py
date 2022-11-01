from rich.progress import track
from torch import optim
from model import *
import torch
import pickle

# -------------------------------------------
max_iter = 2001
lr = 1e-3
r = 10

# -------------load dataset------------------
f = open("2x400x1200x200.bin", "rb")
gt = pickle.load(f)[:, :, :200]
f.close()
f = open("2x400x1200x200_mask.bin", "rb")
MASK = pickle.load(f)
f.close()
train = gt * MASK
Size = train.shape
Train = torch.from_numpy(train)
Gt = torch.from_numpy(gt)

# --------------load model-------------------
mask = torch.ones(Size)
mask[train == 0] = 0
ksam = torch.ones(Size) - MASK  # 用于验证测试集的mask, 预测位置值为1，其余位置为0  117676

# ----------------training-------------------

for r in [20, 30, 40, 50]:
    for gamma in [1e-2, 1e-3, 1e-4, 1e-5]:
        print('gamma:  ', gamma, 'r:  ', r)
        Net = Netlinear1(Size[0], Size[1], Size[2], r=r)
        params = []
        params += [x for x in Net.parameters()]
        optimizier = optim.Adam(params, lr=lr, weight_decay=1e-7)
        for I in track(range(max_iter), description=""):
            Out_real = Net()
            loss = 1e-5 * torch.norm((Out_real - Train) * mask, 2)

            p = params[0]
            loss += gamma * torch.norm(p, 2) * torch.norm(p, 2)
            q = params[1]
            loss += gamma * torch.norm(q, 2) * torch.norm(q, 2)

            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()
            if I % 100 == 0:
                rmse = torch.sqrt(sum(sum(sum(ksam*(Gt-Out_real)*(Gt-Out_real))))/117676)
                if I % 500 == 0:
                    print('iter:  ', I, 'loss:  ', loss.item(), 'rmse:  ', rmse.item())
