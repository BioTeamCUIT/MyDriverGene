import math

import torch
import torch.nn as nn

from torch_geometric.nn import HGTConv, Linear


# Layer:3, hidden_channels: 64
class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        pred = self.lin(x_dict['gene'])

        # out = torch.softmax(input=pred, dim=1)

        return pred

# Layer:3, hidden_channels: 64, Activation: Relu
# class HGT(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
#         super().__init__()
#         self.num_layers = num_layers
#
#         self.lin_dict = torch.nn.ModuleDict()
#         for node_type in data.node_types:
#             self.lin_dict[node_type] = Linear(-1, hidden_channels)
#
#         self.convs = torch.nn.ModuleList()
#         for i in range(num_layers):
#             conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
#                            num_heads, group='sum')
#             self.convs.append(conv)
#
#         self.lin = Linear(hidden_channels, out_channels)
#
#     def forward(self, x_dict, edge_index_dict):
#         x_dict = {
#             node_type: self.lin_dict[node_type](x).relu_()
#             for node_type, x in x_dict.items()
#         }
#
#         for i, conv in enumerate(self.convs):
#             x_dict = conv(x_dict, edge_index_dict)
#             if i != self.num_layers:
#                 # x_dict['gene'] = torch.relu_(x_dict['gene'])
#                 # x_dict['protein'] = torch.relu_(x_dict['protein'])
#                 # x_dict['gene'] = torch.sigmoid(x_dict['gene'])
#                 # x_dict['protein'] = torch.sigmoid(x_dict['protein'])
#                 x_dict['gene'] = torch.tanh(x_dict['gene'])
#                 x_dict['protein'] = torch.tanh(x_dict['protein'])
#
#         return self.lin(x_dict['gene'])

# Layer:3, hidden_channels: 64, MLP: 64/32/16/1
# class HGT(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
#         super().__init__()
#
#         self.lin_dict = torch.nn.ModuleDict()
#         for node_type in data.node_types:
#             self.lin_dict[node_type] = Linear(-1, hidden_channels)
#
#         self.convs = torch.nn.ModuleList()
#         for i in range(num_layers):
#             conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
#                            num_heads, group='sum')
#             self.convs.append(conv)
#
#         # self.lin = Linear(hidden_channels, out_channels)
#
#         self.lin = torch.nn.Sequential(
#             Linear(hidden_channels, 32),
#             # ReLU(inplace=True),
#             nn.Tanh(),
#             Linear(32, 16),
#             # ReLU(inplace=True),
#             nn.Tanh(),
#             Linear(16, out_channels)
#             # ReLU(inplace=True)
#             # nn.Tanh()
#         )
#
#
#
#     def forward(self, x_dict, edge_index_dict):
#         x_dict = {
#             node_type: self.lin_dict[node_type](x).relu_()
#             for node_type, x in x_dict.items()
#         }
#
#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#
#         return self.lin(x_dict['gene'])

# Layer: 3  Hidden_channels:64/32/16
# class HGT(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
#         super().__init__()
#
#         self.lin_dict = torch.nn.ModuleDict()
#         for node_type in data.node_types:
#             self.lin_dict[node_type] = Linear(-1, hidden_channels)
#
#         self.convs = torch.nn.ModuleList()
#         for i in range(num_layers):
#             conv = HGTConv(hidden_channels//pow(2, i), hidden_channels//pow(2, i+1), data.metadata(),
#                            num_heads, group='sum')
#             self.convs.append(conv)
#
#         self.lin = Linear(hidden_channels//pow(2, num_layers), out_channels)
#
#     def forward(self, x_dict, edge_index_dict):
#         x_dict = {
#             node_type: self.lin_dict[node_type](x).relu_()
#             for node_type, x in x_dict.items()
#         }
#
#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#
#         return self.lin(x_dict['gene'])
