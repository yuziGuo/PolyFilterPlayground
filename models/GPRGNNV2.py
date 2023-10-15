import torch
import torch.nn.functional as F
from torch.nn import Linear 
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
import numpy as np

class GPR_propV2(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''
    def __init__(self, K, n_channels, alpha):
        super(GPR_propV2, self).__init__(aggr='add')
        self.K = K
        self.alpha = alpha

        # PPR-like initialization
        TEMP = alpha*(1-alpha)**np.arange(K+1)
        TEMP[-1] = (1-alpha)**K
        TEMP = torch.tensor(TEMP,dtype=torch.float32).unsqueeze(0)
        TEMP = TEMP.repeat(n_channels, 1)
        self.temp = Parameter(TEMP)

    def forward(self, x, edge_index, norm):
        hidden = x*(self.temp[:,0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[:,k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNNV2(torch.nn.Module):
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
                 alpha
                 ):
        super(GPRGNNV2, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        self.act_fn = act_fn

        self.lin1 = Linear(in_feats, n_hidden)
        self.lin2 = Linear(n_hidden, n_classes)

        self.prop1 = GPR_propV2(K, n_hidden, alpha)
        self.dprate = dropout2
        self.dropout = dropout

    # def reset_parameters(self):
    #     self.prop1.reset_parameters()

    def forward(self, feature):
        x = feature

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))

        if self.dprate == 0.0:
            x = self.prop1(x, self.edge_index, self.norm_A)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, self.edge_index, self.norm_A)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        # return F.log_softmax(x, dim=1)

        return x