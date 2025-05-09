import torch
import torchvision

# 方式1 -> 保存方式1，加载模型
model = torch.load("vgg16_method1.pth")
print(model)

# 方式2 -> 保存方式2，加载模型
vgg16 = torchvision.models.vgg16(weights=None)
checkpoint = torch.load("vgg16_method2.pth")
vgg16.load_state_dict(checkpoint)
print(vgg16)
