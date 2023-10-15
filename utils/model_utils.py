from models.NormalBasisNN import NormalNN
from models.NormalBasisNNV2 import NormalNNV2
from models.FavardNormalNN import FavardNormalNN
from models.GPRGNN import GPRGNN
from models.GPRGNNV2 import GPRGNNV2
import torch as th
import torch.nn.functional as F

def build_model(args, edge_index, norm_A, in_feats, n_classes):
    if args.model in ['NormalNN', 'OptBasisGNN']:
        model = NormalNN(
                    edge_index,
                    norm_A,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu, 
                    args.dropout,
                    args.dropout2
                    )
    if args.model in ['NormalNNV2', 'OptBasisGNNV2']:
        model = NormalNNV2(
                    edge_index,
                    norm_A,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu, 
                    args.dropout,
                    args.dropout2
                    )
    if args.model in ['FavardNormalNN', 'FavardGNN']:
        model = FavardNormalNN(
                    edge_index,
                    norm_A,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu, 
                    args.dropout,
                    args.dropout2,
                    )
    if args.model.startswith('GPRGNN'):
        model = globals()[args.model](
                edge_index,
                 norm_A, 
                 in_feats,
                 args.n_hidden,
                 n_classes,
                 args.n_layers,
                 F.relu,
                 args.dropout,
                 args.dropout2,
                 args.alpha
                )
    model.to(args.gpu)
    return model


def build_optimizers(args, model):
    # OptBasis
    if args.model.startswith('OptBasis') or args.model.startswith('Normal') :
        param_groups = [
            {'params':model.fcs.parameters(), 'lr':args.lr1, 'weight_decay':args.wd1}, 
            {'params':[model.alpha_params], 'lr':args.lr2,'weight_decay':args.wd2}
        ]
        optimizer = th.optim.Adam(param_groups)
        return [optimizer]
    
    # FavardGNN
    elif args.model.startswith('Favard'):
        param_groups = [
            {'params':model.fcs.parameters(), 'lr':args.lr1, 'weight_decay':args.wd1}, 
            {'params':[model.alpha_params], 'lr':args.lr2,'weight_decay':args.wd2},
            {'params':[model.yitas, model.sqrt_betas], 'lr':args.lr3,'weight_decay':args.wd3}
        ]
        optimizer = th.optim.Adam(param_groups)
        return [optimizer]
    
    # GPR-GNN
    elif args.model.startswith('GPR'):
        param_groups = [
            {'params':model.lin1.parameters(), 'lr':args.lr1, 'weight_decay':args.wd1}, 
            {'params':model.lin2.parameters(), 'lr':args.lr1, 'weight_decay':args.wd1}, 
            {'params':[model.prop1.temp], 'lr':args.lr2,'weight_decay':args.wd2}
        ]
        optimizer = th.optim.Adam(param_groups)
        return [optimizer]
    

def bce_with_logits_loss(input, labels):
    """
    Input:  Nxc
    target: Nx1
    """
    target = F.one_hot(labels, labels.max()+1).float()
    return F.binary_cross_entropy_with_logits(input, target)


