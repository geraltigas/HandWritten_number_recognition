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
