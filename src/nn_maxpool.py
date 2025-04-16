import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

""" 
最大池化的作用：
    保留输入特征的同时，减小数据量
"""

dataset = torchvision.datasets.CIFAR10("datasets", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

# inputs = torch.tensor([[1, 2, 0, 3, 1],
#                        [0, 1, 2, 3, 1],
#                        [1, 2, 1, 0, 0],
#                        [5, 2, 3, 1, 1],
#                        [2, 1, 0, 1, 1]])
#
# inputs = torch.reshape(inputs, (-1, 1, 5, 5))
# print(inputs.shape)


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, in_put):
        out_put = self.maxpool1(in_put)
        return out_put


module = Module()
# output = module(inputs)
# print(output)
writer = SummaryWriter("logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = module(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
