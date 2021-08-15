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
