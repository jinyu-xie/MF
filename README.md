## [项目链接](https://github.com/jinyu-xie/MF)

## 前言

2006年10月6日，Netflix公司推出了 `Netflix Prize `(2006–2009) ，这个比赛为第一个开发出能够预测电影评分算法（准确度至少比该公司现有的Cinematch系统高10%）的个人或团队提供100万美元的奖金。目前推荐系统中用的最多的是矩阵分解方法,在`Netflix Prize  `推荐系统大赛中取得突出效果。矩阵分解算法是真正意义上的基于模型的协同过滤算法。 通过将用户和标的物嵌入到低维隐式特征空间，获得用户和标的物的特征向量表示，再通过向量的内积来量化用户对标的物的兴趣偏好。



## 数据集

这次实验我们使用了 Netflix Inc 在 Kaggle 上发布的用于比赛的[数据集](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)



## 模型实现

### 基础原理

对于每个用户及电影，使用r个参数进行建模，即每个用户及电影是一个长度为$$r$$的向量，若用$$q_i\in \mathbb{R}^r$$表示用户$i$，若用$$p_j\in \mathbb{R}^r$$表示电影$$j$$，则用户与电影之间的交互或者说是用户对电影的打分可以用两个向量的内积表示，即内积越大，该用户对这部电影的打分会越高，若将用户$$i$$对电影$$j$$的打分记为$$\hat r_{ij}$$，则损失函数可表示为
$$
\mathcal{L}(q_i,p_j) = \sum_{i，j}(r_{ij}-q_i^Tp_j)^2+\lambda_1\sum_i||q_i||_2^2+\lambda_2\sum_j||p_j||_2^2
$$
其中后面的两个范数和是对参数的正则，防止过拟合。实际使用时会给用户以及电影加上一个偏置项，表示某个用户打分偏高或偏低以及电影的质量太高或太差。

对于所有用户以及电影向量可以以矩阵形式储存。对于有$$n$$个用户，$$m$$部电影的数据集来说，可以用$$Q\in \mathbb{R}^{n \times r}, ~P\in \mathbb{R}^{r\times m}$$的矩阵分别表示。则损失函数可以改写为
$$
\mathcal{L}(Q,P) = ||R-(QP+R')||_2^2+\lambda_1||Q||_2^2+\lambda_2||P||_2^2
$$
其中$$R'$$表示偏置。

### 数据集处理

`Netflix Prize `中包含用户480189个，电影17770部；限制于个人服务器的算力，我们对数据集进行裁剪，训练集大小为$$480000\times 17600$$，对于这一过大的矩阵，我们将每$200$部电影划分为一组，将每$$1200$$名用户划分为一组(划分原因仅限于服务器算力，并未进行推理以获得最优划分)，将原矩阵进行重排，可获得大小为$$88\times 400 \times 1200 \times 200$$的张量(tensor)，测试时我们仅选取了该张量的一部分，即大小为$$400 \times 1200 \times 200$$的张量(在服务器上训练速度较快)，该张量的物理解释为有$$400$$组，每组包括$$1200$$名用户对$$200$$部电影的打分。

由于原数据集中并未划分训练集和测试集，我们进行了手工划分。对于上文的$$3$$维张量，其中大概仅$$10W$$个位置中包含用户的打分(非常稀疏)。我们将其$10\%$划分为测试集。我们设计了一个与训练的$$3$$维张量等大的全一张量$$mask$$，将对应训练数据中的有评分处的位置的$$10\%$$置$$0$$，即数据集与$$mask$$对应元素相乘即可获得训练集，数据集与$$01$$互换后的$$mask$$对应元素相乘即可获得测试集。

### 代码

#### 训练模型

```py
class MF(nn.Module):
    def __init__(self, n_1, n_2, n_3, r=10):
        super(MF, self).__init__()
        # 数据集大小
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        # 超参r
        self.r = r
        # 生成可更新的用户矩阵
        self.user = nn.Parameter(torch.Tensor(n_1, n_2, int(self.r)))
		# 初始化参数，可以不用
        self.stdv = 1. / math.sqrt(self.n_1)
        # 生成电影矩阵，用深度学习中的全连接层实现， bais=True表示加上偏置项，可不用
        self.film = nn.Linear(int(self.r), int(n_3), bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.user.data.uniform_(-self.stdv, self.stdv)

    def forward(self):
        # 矩阵乘法，损失函数中的QP+R'
        out = self.film(self.user)
        # 认为将输出归一化到1至5中间，比不加效果要好一些
        out = 1+4*out/torch.max(out)
        return out
```

#### 训练数据

```python
# -------------load dataset------------------
f = gzip.open("1x400x1200x400.bin.gz", "rb")
gt = pickle.load(f)[:, :, 200:400]
f.close()
f = gzip.open("1x400x1200x400_mask.bin.gz", "rb")
MASK = pickle.load(f)[:, :, 200:400]
f.close()
# 训练集
train = gt * MASK
Size = train.shape
Train = torch.from_numpy(train).to(device)
Gt = torch.from_numpy(gt).to(device)

# --------------set mask-------------------
mask = torch.ones(Size).to(device)
mask[train == 0] = 0
 # 用于验证测试集的mask, 预测位置值为1，其余位置为0
ksam = (torch.ones(Size) - MASK).to(device)
Sum = sum(sum(sum(ksam)))
```



#### 训练过程

```python
for r in [10]:
    for lambda in [1e-5]:
        print('lambda:  ', lambda, 'r:  ', r)
        # 随机数种子
        setup_seed(0)
        Net = MF(Size[0], Size[1], Size[2], r=r).to(device)
        # 统计模型参数量
        params = []
        params += [x for x in Net.parameters()]
        # 优化器选用adam
        s = sum([np.prod(list(params.size())) for params in params]);
        print('Number of params: %d' % s)
        
        optimizier = optim.Adam(params, lr=lr, weight_decay=1e-7)
        # 开始训练
        for I in track(range(max_iter)):
            Out_real = Net()
            # 损失函数的各部分，为了易于调参，将两个lambda设置成同样的数
            loss = 1e-5 * torch.norm((Out_real - Train) * mask, 2)
			# 用户矩阵 
            p = params[0]
            loss += lambda * torch.norm(p, 2) * torch.norm(p, 2)
            # 电影矩阵
            q = params[1]
            loss += lambda * torch.norm(q, 2) * torch.norm(q, 2)

            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()
            if I % 100 == 0:
                # 评分指标为rmse
                rmse = torch.sqrt(sum(sum(sum(ksam*(Gt-Out_real)*(Gt-Out_real))))/Sum)
                if I % 200 == 0:
                    print('iter:  ', I, 'loss:  ', loss.item(), 'rmse:  ', rmse.item())
```

#### 测试结果

```
gamma:   1e-05 r:   10
Number of params: 4802200
iter:   0 loss:   0.07588193049057 rmse:   3.4215316988874225
iter:   200 loss:   0.02186709222633042 rmse:   2.1095510441555714
iter:   400 loss:   0.010478964317334975 rmse:   1.0386805919459765
iter:   600 loss:   0.009559971251859098 rmse:   1.017701625457609
iter:   800 loss:   0.0092694776386581 rmse:   1.035366204715832
iter:   1000 loss:   0.00914612299337272 rmse:   1.0462060920801122
iter:   1200 loss:   0.0090382994989288 rmse:   1.0531512438661752
iter:   1400 loss:   0.009118693986674059 rmse:   1.065097378179649
Working... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
```

### 结语

个人调参结果达到的最好效果为1.017，[原论文](https://dl.acm.org/doi/10.1109/MC.2009.263)中在不加时序信息时达到的最佳效果为0.891，距其还有差距，可能的原因个人认为主要是数据集在测试时仅有200个，相比于用户而言过少；以及调参不彻底，未找到最优超参。
