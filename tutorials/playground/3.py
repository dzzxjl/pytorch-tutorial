import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # print(self.conv2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 注意，2D卷积层的输入data维数是 batchsize*channel*height*width

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print('卷积层第一层输出尺寸', x.size())
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print('卷积层第二层输出尺寸：', x.size())

        # view能否理解成减维
        x = x.view(-1, self.num_flat_features(x))
        print('view后尺寸', x.size())
        x = F.relu(self.fc1(x))
        print('全连接层第一层输出尺寸：', x.size())
        x = F.relu(self.fc2(x))
        print('全连接层第二层输出尺寸：', x.size())
        x = self.fc3(x)
        print('全连接层第二层输出尺寸：', x.size())
        print(x)
        return x

    # 计算特征的数量
    def num_flat_features(self, x):
        print('看看这个值', x.size()[0])
        size = x.size()[1:]  # all dimensions except the batch dimension
        print(size)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

for param in net.parameters():
    print('查看参数')
    print(type(param.data), param.size())

# 第一个1是batch dimension
input = Variable(torch.randn(1, 1, 32, 32))
# input = Variable(torch.randn(1, 32, 32))
out = net(input)
print(out)