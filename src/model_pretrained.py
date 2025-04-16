import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("datasets", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
# 修改模型：添加层数
vgg16_true.classifier.add_module('add linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
# 修改模型：更改某一层参数
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
