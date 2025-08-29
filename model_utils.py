'''
Author: zhuqingquan
FilePath: /ModelFactory/model_utils.py
Description: 收集的一些用于构建网络模型的基础模块,module，或者函数
'''
import torch.nn as nn
import torch.nn.functional as F

class hswish(nn.Module):
    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * nn.functional.relu6(x + 3., inplace=self.inplace) / 6.

class hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.relu6(x + 3., inplace=self.inplace) / 6.

class ConvBlock(nn.Module):
    """
    Conv2d + BatchNorm2d + hswish
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = hswish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def ConvBNActivation(in_channels,out_channels,kernel_size,stride,activate):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else hswish(inplace=True)
        )
 
def Conv1x1BNActivation(in_channels,out_channels,activate):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True) if activate == 'relu' else hswish(inplace=True)
        )
 
def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )