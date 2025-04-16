import torch
import torchvision
from PIL import Image

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像预处理
image_path = "imgs/ship.jpg"
image = Image.open(image_path).convert('RGB')
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = transform(image).unsqueeze(0).to(device)  # 添加batch维度并转移到设备

# 加载模型到对应设备
model = torch.load("model_best.pth", map_location=device)
model.eval()

# 推理
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
