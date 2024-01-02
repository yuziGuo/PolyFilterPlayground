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

class GPRGNNV2Augmented(torch.nn.Module):
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
        super(GPRGNNV2Augmented, self).__init__()
        self.edge_index = edge_index
        self.edge_index2 = edge_index2
        self.norm_A = norm_A
        self.norm_A_2 = norm_A_2

        self.act_fn = act_fn

        self.lin1_1 = Linear(in_feats, n_hidden//2)
        self.lin1_2 = Linear(in_feats, n_hidden//2)
        self.lin2 = Linear(n_hidden, n_classes)
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(self.lin1_1)
        self.fcs.append(self.lin1_2)
        self.fcs.append(self.lin2)

        self.prop1 = GPR_propV2(K, n_hidden//2, alpha)
        self.prop2 = GPR_propV2(K, n_hidden//2, alpha)
        self.dprate = dropout2
        self.dropout = dropout

    # def reset_parameters(self):
    #     self.prop1.reset_parameters()

    def forward(self, feature):
        x = feature

        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = F.relu(self.lin1_1(x))
        x2 = F.relu(self.lin1_2(x))
        
        if self.dprate == 0.0:
            x1 = self.prop1(x1, self.edge_index, self.norm_A)
            x2 = self.prop2(x2, self.edge_index2, self.norm_A_2)
        else:
            x1 = F.dropout(x1, p=self.dprate, training=self.training)
            x1 = self.prop1(x1, self.edge_index, self.norm_A)
            x2 = F.dropout(x2, p=self.dprate, training=self.training)
            x2 = self.prop2(x2, self.edge_index2, self.norm_A_2)
        
        x = torch.hstack((x1,x2))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x