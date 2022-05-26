import torch
# 这一例子仅可用于每个op只产生一个输出的情况，且效率很低（由于对于某一节点，每次未等待所有梯度反传至此节点，就直接将本次反传回的梯度直接反传至叶节点）
def autograd(grad_fn, gradient):
    auto_grad = {}
    queue = [[grad_fn, gradient]]
    while queue != []:
        item = queue.pop()
        gradients = item[0](item[1])
        functions = [x[0] for x in item[0].next_functions]
        if type(gradients) is not tuple:
            gradients = (gradients,)
        for grad, func in zip(gradients, functions):
            if type(func).__name__ == 'AccumulateGrad':
                if hasattr(func.variable, 'auto_grad'):
                    func.variable.auto_grad = func.variable.auto_grad + grad
                else:
                    func.variable.auto_grad = grad
            else:
                queue.append([func, grad])


A = torch.tensor([3.], requires_grad=True)
B = torch.tensor([2.], requires_grad=True)
C = A ** 2
D = B ** 2
E = C * D
F = D + E

autograd(F.grad_fn, torch.tensor(1)) #F.backward()
print(A.auto_grad, B.auto_grad)  # tensor(24., grad_fn=<UnbindBackward>) tensor(40., grad_fn=<AddBackward0>)

# 这一autograd同样可作用于编写的模型，我们将会看到，它与pytorch自带的backward产生了同样的结果
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)
        self.fc3 = nn.Linear(5, 2)
        self.fc4 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x1 = self.fc2(x)
        x2 = self.fc3(x)
        x2 = self.relu(x2)
        x2 = self.fc4(x2)
        return x1 + x2


x = torch.ones([10], requires_grad=True)
mlp = MLP()
mlp_state_dict = mlp.state_dict()
# 自定义autograd
mlp = MLP()
mlp.load_state_dict(mlp_state_dict)
y = mlp(x)
z = torch.sum(y)
autograd(z.grad_fn, torch.tensor(1.))
print(
    x.auto_grad)  # tensor([-0.0121,  0.0055, -0.0756, -0.0747,  0.0134,  0.0867, -0.0546,  0.1121, -0.0934, -0.1046], grad_fn=<AddBackward0>)
mlp = MLP()
mlp.load_state_dict(mlp_state_dict)
y = mlp(x)
z = torch.sum(y)
z.backward()
print(x.grad)  # tensor([-0.0121,  0.0055, -0.0756, -0.0747,  0.0134,  0.0867, -0.0546,  0.1121, -0.0934, -0.1046])