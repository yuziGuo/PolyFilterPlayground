'''
Borrowed from
https://github.com/jianhao2016/GPRGNN
'''

import torch
import torch.nn.functional as F
from torch.nn import Linear
from layers.GPRConv import GPR_prop

class GPRGNNAugmented(torch.nn.Module):
    def __init__(self,
                 edge_index,
                 edge_index2,
                 norm_A, 
                 norm_A_2,
                 in_feats,
                 n_hidden,
                 n_classes,
                 K,
                 act_fn,
                 dropout,
                 dropout2,
                 alpha
                 ):
        super(GPRGNNAugmented, self).__init__()
        self.edge_index = edge_index
        self.edge_index2 = edge_index2
        self.norm_A = norm_A
        self.norm_A_2 = norm_A_2
        self.act_fn = act_fn

        self.lin1 = Linear(in_feats, n_hidden)
        self.lin2 = Linear(n_hidden, n_classes)

        self.fcs = torch.nn.ModuleList()
        self.fcs.append(self.lin1)
        self.fcs.append(self.lin2)

        self.prop1 = GPR_prop(K, alpha)
        self.prop2 = GPR_prop(K, alpha)
        self.dprate = dropout2
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, feature):
        x = feature

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        if self.dprate == 0.0:
            x1 = self.prop1(x, self.edge_index, self.norm_A)
            x2 = self.prop2(x, self.edge_index2, self.norm_A_2)
            return x1 + x2
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x1 = self.prop1(x, self.edge_index, self.norm_A)
            x2 = self.prop2(x, self.edge_index2, self.norm_A_2)
            return x1 + x2