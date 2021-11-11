import torch
from torch import nn

from modules import ResGCN_Module


class ResGCN_Input_Branch(nn.Module):
    def __init__(self, structure, block, num_channel, A, **kwargs):
        super(ResGCN_Input_Branch, self).__init__()

        self.register_buffer('A', A)
        module_list = [ResGCN_Module(num_channel, 64, 'Basic', A, initial=True, **kwargs)]
        module_list += [ResGCN_Module(64, 64, 'Basic', A, initial=True, **kwargs) for _ in range(structure[0] - 1)]
        module_list += [ResGCN_Module(64, 64, block, A, **kwargs) for _ in range(structure[1] - 1)]
        module_list += [ResGCN_Module(64, 32, block, A, **kwargs)]

        self.bn = nn.BatchNorm2d(num_channel)
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):

        N, C, T, V, M = x.size()
        x = self.bn(x.permute(0,4,1,2,3).contiguous().view(N*M, C, T, V))
        for layer in self.layers:
            x = layer(x, self.A)

        return x