import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.nn import global_add_pool, global_mean_pool, GCNConv

from functools import partial


class ResGCN(torch.nn.Module):
    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool='sum', dropout=0,
                 edge_norm=True):
        super(ResGCN, self).__init__()
        self.conv_residual = residual
        self.fc_residual = False
        self.res_branch = res_branch
        self.collapse = collapse
        self.global_pool = global_add_pool
        self.dropout = dropout

        # GConv = GCNConv(in_channels=)

        # hidden_in = dataset.num_features
        hidden_in = 48

        self.bn_feature = BatchNorm1d(num_features=hidden_in)
        feat_gfn = True
        self.conv_feature = GCNConv(in_channels=hidden_in, out_channels=hidden)

        self.proj_head = nn.Sequential(nn.Linear(48, 48), nn.ReLU(inplace=True), nn.Linear(48, 48))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        return self.forward_BNConvReLU(x, edge_index)

    def forward_BNConvReLU(self, x, edge_index):
        x = self.bn_feature(x)
        x = F.relu(self.conv_feature(x, edge_index))
