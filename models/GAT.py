'''From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py
'''

import torch
import torch.nn.functional as F
from layers.GATConv import GATConv        

class GAT(torch.nn.Module):
    def __init__(self, 
            edge_index,
            norm_A,
            in_channels, 
            hidden_channels, 
            out_channels, 
            n_layers,
            heads,
            out_heads=1,
            dropout=0.6,
            ):
        super().__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        self.n_layers = n_layers
        self.dropout=dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels//heads, heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=out_heads, concat=False, dropout=dropout))


    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[0](x, self.edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, self.edge_index)
        # x = F.log_softmax(x, dim=1)
        return x

        