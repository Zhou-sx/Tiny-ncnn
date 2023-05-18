import torch
import torch.nn as nn

a = torch.zeros(1,9,9)
b = torch.ones(1,9,9)
c = torch.ones(1,9,9)

input = torch.unsqueeze(torch.concat([a, b, c], axis = 0), axis = 0)

m = nn.Conv2d(3, 5, (3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), bias=False)

conv_weight = torch.arange(0, 3*3*3*5, 1).float() # 先创建一个自定义权值的Tensor
conv_weight = conv_weight.reshape((5, 3, 3, 3))

m.weight=torch.nn.Parameter(conv_weight)
# print(m.weight)

output = m(input)
print(output)