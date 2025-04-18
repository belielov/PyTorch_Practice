import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("datasets", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
module = Module()  # 模型实例化
optim = torch.optim.SGD(module.parameters(), lr=0.01)  # 定义优化器

for epoch in range(20):
    running_loss = 0.0  # 统计每个训练轮次的损失
    for data in dataloader:
        imgs, targets = data
        outputs = module(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()  # 清零梯度
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print(running_loss)
