import torch.nn as nn
from layers.NormalBasisConv import NormalBasisConv

import torch as th
import torch.nn.functional as F

class NormalNNAugmented(nn.Module):
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
                 ):
        super(NormalNNAugmented, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A

        self.edge_index2 = edge_index2
        self.norm_A_2 = norm_A_2
        
        self.convs = nn.ModuleList()
        for _ in range(K):
            self.convs.append(NormalBasisConv())
        self.K = K
        self.n_channel = n_hidden//2
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feats, n_hidden//2))
        self.fcs.append(nn.Linear(in_feats, n_hidden//2))
        self.fcs.append(nn.Linear(n_hidden, n_classes))
        self.act_fn = act_fn
        
        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout2)

        self.init_alphas()
    
    def init_alphas(self):
        t = th.zeros(self.K+1)
        t[0] = 1
        t = t.repeat(self.n_channel, 1)

        self.alpha_params1 = nn.Parameter(t.float()) 
        self.alpha_params2 = nn.Parameter(t.float()) 

        # self.alpha_params = th.stack([self.alpha_params1, self.alpha_params2])
        self.alpha_params = [self.alpha_params1, self.alpha_params2]
        # self.alpha_params = nn.ModuleList()
        # self.alpha_params.append(self.alpha_params1)
        # self.alpha_params.append(self.alpha_params2)
        


    def forward(self, features):
        x = features

        x = self.dropout(x)
        x1 = self.fcs[0](x)
        x2 = self.fcs[1](x)
        
        x1 = self.act_fn(x1)  
        x1 = self.dropout2(x1)

        x2 = self.act_fn(x2)  
        x2 = self.dropout2(x2) 

        blank_noise = th.randn_like(x1)*1e-5
        x1 = x1 + blank_noise
        blank_noise = th.randn_like(x2)*1e-5
        x2 = x2 + blank_noise
        

        rsts = []
        for _, x in enumerate([x1, x2]):
            h0 = x / th.clamp((th.norm(x,dim=0)), 1e-8)
            rst = th.zeros_like(h0)
            rst = rst + self.alpha_params[_][:,0] * h0

            last_h = h0
            second_last_h = th.zeros_like(h0)

            for i, con in enumerate(self.convs, 1):
                h_i = con(self.edge_index, self.norm_A, last_h, second_last_h)
                rst = rst + self.alpha_params[_][:,i] * h_i
                second_last_h = last_h
                last_h = h_i
            
            rsts.append(rst)
        
        rst = th.hstack(rsts)
        rst = self.dropout(rst)
        rst = self.fcs[-1](rst)
        # rst = F.log_softmax(rst, dim=1)
        return rst