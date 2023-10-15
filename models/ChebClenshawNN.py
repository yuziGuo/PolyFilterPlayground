import imp
import torch.nn as nn
import torch.nn.init as init
from layers.ChebClenshawConv import ChebConv

import torch as th
import torch.nn.functional as F

import math


def relu(x):
    """This implementation allows subgradient of relu(x) at x = 0 to be 1 instead of 0.
    """
    x[x < -0.0] = 0
    return x


class ChebNN(nn.Module):
    def __init__(
        self,
        edge_index,
        norm_A,
        in_feats,
        n_hidden,
        n_classes,
        K,
        dropout,
        dropout2,
        lamda,
        dropW=False, 
        dropAct=False
    ):  
        super(ChebNN, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A

        self.convs = nn.ModuleList()
        for _ in range(K + 1):
            self.convs.append(
                ChebConv(n_hidden, n_hidden, K, lamda=lamda, weight=not dropW, bias=not dropW)
            )
        self.K = K
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feats, n_hidden))
        self.fcs.append(nn.Linear(n_hidden, n_classes))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())

        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout2)
        self.init_alphas()

        self.dropAct = dropAct

    def init_alphas(self):
        '''
        Author's note: 
            Combine Line84 to see that, this initialization is equivalent with setting the initial underlying filtering polynomial to be h(\lambda)=1-\lambda 
            (\lamba is an eigenvalue of the laplacian matrix)
        '''
        t = th.zeros(self.K + 1)
        t[0] = 1
        self.alpha_params = nn.Parameter(t.float())

    def forward(self, features):
        x = features

        x = self.dropout(x)
        x = self.fcs[0](x)
        x = relu(x)

        x = self.dropout(x)
        h0 = x
        last_h = th.zeros_like(h0)
        second_last_h = th.zeros_like(h0)
        
        for i, con in enumerate(self.convs):
            '''
            Authors' note: 
                Note that the order of alpha params is INVERSED in clenshaw 
                algorithm. (Check Theorem 3.1)
            '''
            alpha = self.alpha_params[-(i + 1)]
            x = con(self.edge_index, self.norm_A, h0, last_h, second_last_h, alpha, i)
            if not self.dropAct:
                if i < self.K - 1:
                    x = relu(x)
                    x = self.dropout2(x) 
            second_last_h = last_h
            last_h = x

        x = relu(x)
        x = self.dropout(x)
        x = self.fcs[-1](x)

        # x = F.log_softmax(x, dim=1)
        return x
