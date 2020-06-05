
import torch

import torch.nn as nn

net = nn.LSTM(3,4,1, bidirectional=True, batch_first=True)  # 隐层尺度为4

x = torch.rand(1, 5, 3)  # 序列长度为5，输入尺度为3


# 去掉一些右边信息，但左起信息保留

o=net(x)  # 从结果比较，可以看出concat顺序是先正向再反向，h_n正向部分是正向最后计算结果

x2 = x[:, 2:, :]  # 去掉一些左边信息，但右起信息保留

o1=net(x2)  # 通过结果比较，发现concat是按照实际序列位置进行的，且h_n的反向部分是反向最后计算结果
print()
