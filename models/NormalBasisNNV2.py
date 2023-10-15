import torch.nn as nn
from layers.NormalBasisConv import NormalBasisConv

import torch as th
import torch.nn.functional as F


class NormalNNV2(nn.Module):
    '''
    class NormalNNV2 is just slightly different to class NormalNN
    '''
    def __init__(self,
                 edge_index,
                 norm_A, 
                 in_feats,
                 n_hidden,
                 n_classes,
                 K,
                 act_fn,
                 dropout,
                 dropout2,
                 ):
        super(NormalNNV2, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        
        self.convs = nn.ModuleList()
        for _ in range(K):
            self.convs.append(NormalBasisConv())
        self.K = K
        self.n_channel = n_classes
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feats, n_hidden))
        self.fcs.append(nn.Linear(n_hidden, n_classes))
        self.act_fn = act_fn
        
        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout2)

        self.init_alphas()
    
    def init_alphas(self):
        t = th.zeros(self.K+1)
        t[0] = 1
        t = t.repeat(self.n_channel , 1)
        self.alpha_params = nn.Parameter(t.float()) 


    def forward(self, features):
        x = features

        x = self.dropout(x)
        x = self.fcs[0](x) 
        x = self.act_fn(x) 
        
        x = self.dropout(x) 
        x = self.fcs[1](x) 

        x = self.dropout2(x)

        # No need for blank noise in this version
        # TODO: why
        # blank_noise = th.randn_like(x)*1e-5
        # x = x + blank_noise
        
        h0 = x / th.clamp((th.norm(x,dim=0)), 1e-8)
        rst = th.zeros_like(h0)
        rst = rst + self.alpha_params[:,0] * h0

        last_h = h0
        second_last_h = th.zeros_like(h0)
        for i, con in enumerate(self.convs, 1):
            h_i = con(self.edge_index, self.norm_A, last_h, second_last_h)
            rst = rst + self.alpha_params[:,i] * h_i
            second_last_h = last_h
            last_h = h_i
        
        # rst = F.log_softmax(rst, dim=1)
        return rst