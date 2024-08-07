import torch as th
from torch_geometric.nn import MessagePassing

class OptBasisConv(MessagePassing):
    def __init__(self):
        super(OptBasisConv, self).__init__()

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, edge_index, norm_A, last_h, second_last_h):
        rst = self.propagate(edge_index=edge_index, x=last_h, norm=norm_A)
        _t = th.einsum('nh,nh->h',rst,last_h)
        rst = rst - th.einsum('h,nh->nh', _t, last_h)
        _t = th.einsum('nh,nh->h',rst,second_last_h)
        rst = rst - th.einsum('h,nh->nh', _t, second_last_h)
        rst = rst / th.clamp((th.norm(rst,dim=0)),1e-8)
        return rst
