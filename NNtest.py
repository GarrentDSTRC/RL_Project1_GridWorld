import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
# 创建一个很简单的网络：两个卷积层，一个全连接层
class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1, bias=False)
        self.linear = nn.Linear(32*10*10, 20, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.linear(x.view(x.size(0), -1))
        return x

x=torch.tensor([1.,2.],requires_grad=True)
#x2=torch.tensor([2.],requires_grad=True)
y=0.1*x[0]+0.2*x[1]
y.retain_grad()
z=(y-1)**2
z.backward()
print(z.sum(),y.grad,x.grad)


model = Simple()
# 为了方便观察数据变化，把所有网络参数都初始化为 0.1
for m in model.parameters():
    m.data.fill_(0.1)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

model.train()
# 模拟输入8个 sample，每个的大小是 10x10，
# 值都初始化为1，让每次输出结果都固定，方便观察
images = torch.ones(8, 3, 10, 10)
targets = torch.ones([8,20], dtype=torch.long)

output = model(images)
print(output.shape)
# torch.Size([8, 20])

#loss = criterion(output, targets)
loss = ((output-targets)**2).mean()

print(model.conv1.weight.grad)
# None
loss.backward()
print(model.linear.weight.grad)
# tensor([-0.0782, -0.0842, -0.0782])
# 通过一次反向传播，计算出网络参数的导数，
# 因为篇幅原因，我们只观察一小部分结果

print(model.conv1.weight[0][0][0])
# tensor([0.1000, 0.1000, 0.1000], grad_fn=<SelectBackward>)
# 我们知道网络参数的值一开始都初始化为 0.1 的

optimizer.step()
print(model.conv1.weight[0][0][0])
# tensor([0.1782, 0.1842, 0.1782], grad_fn=<SelectBackward>)
# 回想刚才我们设置 learning rate 为 1，这样，
# 更新后的结果，正好是 (原始权重 - 求导结果) ！

optimizer.zero_grad()
print(model.conv1.weight.grad[0][0][0])
# tensor([0., 0., 0.])
# 每次更新完权重之后，我们记得要把导数清零啊，
# 不然下次会得到一个和上次计算一起累加的结果。
# 当然，zero_grad() 的位置，可以放到前边去，
# 只要保证在计算导数前，参数的导数是清零的就好。