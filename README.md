# PyTorch_Practice
这里是学习pytorch框架过程中的笔记
## NumPy数组转换为PyTorch张量
### 方法
```python
true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32)
    for x in [true_w, features, poly_features, labels]
]

# 等价于

true_w = torch.tensor(true_w, dtype=torch.float32)
features = torch.tensor(features, dtype=torch.float32)
poly_features = torch.tensor(poly_features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
```
### 原因
1. 框架要求：
- PyTorch的神经网络层（如 `nn.Linear` ）要求输入是张量（tensor）
- NumPy数组不能直接用于PyTorch模型的前向传播
2. GPU加速：
- 张量可以转移到GPU（ `.to('cuda')` ）进行加速计算
- NumPy数组只能在CPU上运行
3. 自动求导：
- 张量可以设置 `requires_grad=True` 启用自动微分
- NumPy数组没有梯度计算能力
4. 数据一致性：
- 确保所有数据使用相同的精度（float32）
- 避免混合精度导致的意外行为