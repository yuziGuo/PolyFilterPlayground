import torch as th
import torch.nn.functional as F

from models import *

def build_model(args, edge_index, edge_weights, in_feats, n_classes):
    if args.model.startswith(('OptBasis',  'Favard', 'BernNet', 'ChebNetII')):
        model = globals()[args.model](
                    edge_index,
                    edge_weights,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu, 
                    args.dropout,
                    args.dropout2
                    )
        
    elif args.model in ['GPRGNN', 'GPRGNNV2']:
        model = globals()[args.model](
                edge_index,
                 edge_weights, 
                 in_feats,
                 args.n_hidden,
                 n_classes,
                 args.n_layers,
                 F.relu,
                 args.dropout,
                 args.dropout2,
                 args.alpha
                )
        
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    model.to(args.gpu)
    
    return model


def build_optimizers(args, model):
    if args.model.startswith('OptBasis') :
        param_groups = [
            {'params':model.fcs.parameters(), 'lr':args.lr1, 'weight_decay':args.wd1}, 
            {'params':[model.alpha_params], 'lr':args.lr2,'weight_decay':args.wd2}
        ]
        optimizer = th.optim.Adam(param_groups)
    
    elif args.model.startswith('Favard'):
        param_groups = [
            {'params':model.fcs.parameters(), 'lr':args.lr1, 'weight_decay':args.wd1}, 
            {'params':[model.alpha_params], 'lr':args.lr2,'weight_decay':args.wd2},
            {'params':[model.gammas, model.sqrt_betas], 'lr':args.lr3,'weight_decay':args.wd3}
        ]
        optimizer = th.optim.Adam(param_groups)
    
    elif args.model.startswith(('GPR', 'Cheb', 'Bern')):
        param_groups = [
            {'params':model.lin1.parameters(), 'lr':args.lr1, 'weight_decay':args.wd1}, 
            {'params':model.lin2.parameters(), 'lr':args.lr1, 'weight_decay':args.wd1}, 
            {'params':[model.prop1.temp], 'lr':args.lr2,'weight_decay':args.wd2}
        ]
        optimizer = th.optim.Adam(param_groups)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    return [optimizer]
    

def bce_with_logits_loss(input, labels):
    """
    Input:  Nxc
    target: Nx1
    """
    target = F.one_hot(labels, labels.max()+1).float()
    return F.binary_cross_entropy_with_logits(input, target)


