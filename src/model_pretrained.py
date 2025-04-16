import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("datasets", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
# �޸�ģ�ͣ���Ӳ���
vgg16_true.classifier.add_module('add linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
# �޸�ģ�ͣ�����ĳһ�����
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
