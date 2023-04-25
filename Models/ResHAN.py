import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.nn import global_add_pool, global_mean_pool, HANConv

from functools import partial


class ResHAN(torch.nn.Module):
    def __init__(self, metadata, in_channels, out_channels, hidden_size=128, heads=8):
        super(ResHAN, self).__init__()
        self.han_conv = HANConv(in_channels=in_channels, out_channels=hidden_size, heads=heads, dropout=0.6,
                                metadata=metadata)
        self.lin = nn.Linear(in_features=hidden_size, out_features=out_channels)

        self.proj_head = nn.Sequential(nn.Linear(hidden_size, in_channels), nn.ReLU(inplace=True), nn.Linear(hidden_size, in_channels))

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['gene'])
        return

    def forward_cl(self,x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['gene'])
        out = self.proj_head(out['gene'])
        return out

    def loss_cl(self, x1, x2):
        T = 0.5
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()


