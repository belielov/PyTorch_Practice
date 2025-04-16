import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("训练设备：", device)

# 准备数据集
train_data = torchvision.datasets.CIFAR10("datasets", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("datasets", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# 查看数据集
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
model = Model()
model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
model.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")

# 记录开始时间
start_time = time.time()

for i in range(epoch):
    print("------------- 第 {} 轮训练开始 -------------".format(i+1))

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练时间：{:.4f}s, 次数：{}, Loss：{:.4f}".format(end_time - start_time, total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的准确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(model, "model_{}.pth".format(i))  # 方式1：模型结构及其参数
    # torch.save(model.state_dict(), "model_{}.pth".format(i))  # 方式2：仅模型参数
    print("模型已保存")

writer.close()
