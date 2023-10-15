import torch as th
import torch.nn as nn
import torch.nn.init as init
import math

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GraphConvII(MessagePassing):
    def __init__(self,
                 in_feats,
                 out_feats,
                 ):
        super(GraphConvII, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(th.Tensor(out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self.bias)
        stdv = 1. / math.sqrt(self._out_feats)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, norm_A, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        rst = self.propagate(edge_index, x=x, norm=norm_A)
        rst = (1 - alpha) * rst + alpha * h0   

        rstw  = th.matmul(rst, self.weight) + self.bias
        rst = theta * rstw + (1 - theta) * rst
        return rst
