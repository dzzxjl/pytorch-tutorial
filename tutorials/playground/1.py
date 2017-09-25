import torch

# 生成 3*4 的矩阵，未初始化
a = torch.IntTensor(3, 4)
print(a)
# 生成 3*4 的矩阵，初始化
b = torch.rand(3, 4)
print(b)
# 输出b的大小
print(b.size)

c = torch.rand(3, 4)

d = b + c
print(d)

# python中二维进行切片操作 ','前代表对行切片 ','后代表对列切片
print(d[:2,1])

e = torch.ones(5)
print(e)

f = e.numpy()
print(f)