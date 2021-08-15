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
