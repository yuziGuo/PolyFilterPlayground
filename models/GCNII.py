import torch.nn as nn
from layers.GCNIIConv import GraphConvII

import torch as th
import torch.nn.functional as F

class GCNII(nn.Module):
    def __init__(self,
                 edge_index,
                 norm_A,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 alpha,
                 lamda,
                 ):
        super(GCNII, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GraphConvII(n_hidden, n_hidden))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feats, n_hidden))
        self.fcs.append(nn.Linear(n_hidden, n_classes))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = F.relu
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = alpha
        self.lamda = lamda

    def predict(self, features):
        with th.no_grad():
            self.eval()
            x = features
            x = self.dropout(x)
            x = self.fcs[0](x)
            x = self.act_fn(x)
            h0 = x

            for i, con in enumerate(self.convs):
                x = self.dropout(x)
                x = con(x, self.edge_index, self.norm_A, 
                        h0, self.lamda, self.alpha, i+1)
                x = self.act_fn(x)

            x = self.dropout(x)
            x = self.fcs[-1](x)
            return x

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.fcs[0](x)
        x = self.act_fn(x)
        h0 = x

        for i, con in enumerate(self.convs):
            x = self.dropout(x)
            x = con(x, self.edge_index, self.norm_A, 
                    h0, self.lamda, self.alpha, i+1)
            x = self.act_fn(x)

        x = self.dropout(x)
        x = self.fcs[-1](x)
        # x = F.log_softmax(x, dim=1)
        return x



