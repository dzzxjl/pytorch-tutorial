import torch.nn
from torch import nn
from torch import autograd

# m = nn.Linear(20, 30)
# input = autograd.Variable(torch.randn(128, 20))
# output = m(input)
# print(output)
m = nn.Dropout2d(p=0.2)
input = autograd.Variable(torch.randn(20, 16, 32, 32))
output = m(input)
for param in m.parameters():
    print("!!!!!!!!!!")
    print(param.data)
print(output)