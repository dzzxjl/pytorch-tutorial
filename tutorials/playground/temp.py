import torch
from torch.autograd import Variable



a = Variable(torch.randn(2, 1, 3, 3))
print(a)