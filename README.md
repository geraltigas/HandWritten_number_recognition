
# MNIST 手写数字识别 Pytorch CNN AI


这前两天抽时间把pytorch入门看完了,心潮澎湃.尽管我不是很了解CNN训练的原理,但是这并不妨碍我去使用它(滑稽

这次训练的 CNN 是用来 识别0~9的手写数字, 输入为 单通道的28*28([1,28,28] tensor) 图片, 输出为 [10] tensor

使用的广为流传的图片分类CNN模型LeNET模型, [论文](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)上的原图中的subsampling层没看懂,但是我看别人都是用maxpooling来代替的,但实际上我查了查还是有很大的差别的.虽然效果最后不错,这可能就是神经网络的鲁棒性很强吧(呵呵呵),就像人的脑子少了某些地方也不影响正常使用. [github仓库](https://github.com/geraltigas/HandWritten_number_recognition)

**遇到的困难:**

- 32\*32 怎么就变成了28*28,为了使模型大致保持论文中的样子,我强行加了个padding.不过好好想也想得通,因为要识别准确的话,必须要训练出能够在等比放大缩小的情况下还能分辨出数字的网络,所以说如果28变32不行,那32也别想识别更小的了
- 我除了这上面用了的layer,其他的是一 窍 不 通

``` python
terminal:
开始测试
测试结束,loss_total: 514.0682983398438
正确率: 98.39%
```

![论文中的模型图](https://raw.githubusercontent.com/geraltigas/image/master/CNN_LeNET.png)


**模型类**

``` python
# nn_module for MNIST
# reference: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
# reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

from torch import nn


class MNIST_model(nn.Module):
    def __init__(self):
        super(MNIST_model, self).__init__()
        self.nn_net = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),  # 论文是32*32???,但是我只有28*28
            nn.MaxPool2d(2, 2),  # subsampling 是什么???我干脆用pooling代替了
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),  # 为全连接层做准备
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, input):
        output = self.nn_net(input)
        return output
```

**训练代码**

``` python
# description: 利用MINST数据集实现手写数字识别
import torch
import torchvision
from torch.utils.data import DataLoader
from nn_module import MNIST_model
import time

# MACRO setting
EPOCH = 20
BATCH_SIZE = 100
LR = 1e-2
# device init
device = torch.device("cuda")

# 第一步:获得数据集
# reference: pytorch.org


train_dataset = torchvision.datasets.MNIST(
    root='./data_MNIST',
    transform=torchvision.transforms.ToTensor(),
    train=True,
    download=True
)

print(train_dataset.data.size())

# 第二步:加载数据集进入dataloader
# reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

data_load_train = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,  # 指定每次投喂(???)的图片的数量
    shuffle=True,  # 每个epoch重新打乱顺序
)

# 第三步:引入神经网络模型,实例化.创建optimizer,loss_function
ins_model = MNIST_model().cuda()
loss_func = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(ins_model.parameters(), lr=LR)

time_start = time.time()

# 第四步:投喂数据,
for epoch_count in range(EPOCH):
    print('this is epoch {}'.format(epoch_count + 1))
    loss_for_the_epoch = 0
    for sing_data in data_load_train:
        img, target = sing_data
        img = img.to(device)  # 将数据转移到GPU计算
        target = target.to(device)  # 将数据转移到GPU计算
        output = ins_model(img)
        loss = loss_func(output, target)
        loss_for_the_epoch += loss
        # 根据loss来优化神经网络
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 向后传播,更新梯度
        optimizer.step()  # 根据梯度优化网络参数

    print('epoch {},loss_total = {}'.format(epoch_count, loss_for_the_epoch))

time_end = time.time()
print('总耗时', time_end - time_start, 's')

torch.save(ins_model, "./model/MNIST.pth")

print("\n\n ------------ \n训练结束,开始测试")
```

**测试代码**

``` python
import torch
import torchvision
from torch.utils.data import DataLoader

from nn_module import MNIST_model

EPOCH = 20
BATCH_SIZE = 1
LR = 1e-2

device = torch.device('cuda')

test_dataset = torchvision.datasets.MNIST(
    root='./data_MNIST',
    transform=torchvision.transforms.ToTensor(),
    train=False,
    download=True
)

print(test_dataset.data.size())

data_load_test = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

ins_model = torch.load("./model/MNIST.pth").cuda()

data_total = 0
correct = 0
loss_test = 0

loss_func = torch.nn.CrossEntropyLoss().cuda()
print('开始测试')

for data in data_load_test:
    data_total += 1
    img, target = data
    img = img.to(device)
    target = target.to(device)
    output = ins_model(img)
    loss_test += loss_func(output, target)
    if output.argmax(1).item() == target.item():
        correct += 1

print('测试结束,loss_total: {}'.format(loss_test))

print('正确率: {}%'.format((correct / data_total) * 100))
```