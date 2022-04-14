"""
@ModuleName: test
@Description: 
@Author: Beier
@Time: 2022/4/184 18:21
"""

from torch import nn
import torch

tensor1 = torch.ones(15, 768, 10, 10)
conv1 = nn.Conv1d(in_channels=768, out_channels=786, kernel_size=3)
print(conv1(tensor1).shape)
