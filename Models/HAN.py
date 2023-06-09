import os

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import HANConv


class HAN(nn.Module):
    def __init__(self, metadata, in_channels, out_channels, hidden_size=64, heads=4):
        super(HAN, self).__init__()
        self.han_conv = HANConv(in_channels=in_channels, out_channels=hidden_size, heads=heads, dropout=0.2,
                                metadata=metadata)
        self.lin = nn.Linear(in_features=hidden_size, out_features=out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict['protein'] = (x_dict['protein'] - x_dict['protein'].min()) / (
                    x_dict['protein'].max() - x_dict['protein'].min())
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['gene'])
        return out
