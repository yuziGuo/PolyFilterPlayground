from re import L
from networkx.algorithms.shortest_paths import weighted
from numpy import unicode_, var
import torch as th
from torch import Tensor
from torch_sparse import SparseTensor
from torch.functional import norm
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing

class ChebConv(MessagePassing):
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_layers,
                 lamda=1,
                 weight=True,
                 bias=False
                 ):
        super(ChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.lamda = lamda

        self.n_layers =  n_layers
        self.thetas = th.log(lamda / (th.arange(n_layers)+1) + 1)
        _ones = th.ones_like(self.thetas)
        self.thetas = th.where(self.thetas<1, self.thetas, _ones)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            init.zeros_(self.bias)
        if self.weight is not None:
            stdv = 1. / math.sqrt(self._out_feats)
            self.weight.data.uniform_(-stdv, stdv)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, edge_index, norm_A, h0, last_h, second_last_h, alpha, l):
        rst = self.propagate(edge_index=edge_index, x=last_h, norm=norm_A)
        rst = alpha * h0 + 2*rst - second_last_h
        theta = self.thetas[l-1]
        rstw = rst
        weight = self.weight
        if weight is not None:
            rstw  = th.matmul(rst, weight)  
        if self.bias is not None:
            rstw  = rstw + self.bias
        rst = theta * rstw + (1 - theta) * rst 
        return rst
