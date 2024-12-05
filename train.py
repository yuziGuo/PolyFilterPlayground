import argparse
import random
import time 

import numpy as np
import torch as th
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import seaborn as sns

from utils.data_utils import build_dataset
from utils.model_utils import build_model, build_optimizers
from utils.grading_logger import _set_logger
from utils.random_utils import reset_random_seeds
from utils.stopper import EarlyStopping
from utils.rocauc_eval import eval_rocauc
from utils.model_utils import bce_with_logits_loss
from utils.rocauc_eval import fast_auc_th, fast_auc, acc

def build_stopper(args):
    stopper = EarlyStopping(patience=args.patience, store_path=args.es_ckpt+'.pt', gauge='acc')
    step = stopper.step
    return step, stopper


def evaluate(logits, labels, mask, evaluator):    
    if not th.is_tensor(logits):
        logits = logits[0]
    logits = logits[mask]
    labels = labels[mask]
    metric = evaluator(logits, labels)
    return metric


def run(args, cv_id, edge_index, data, norm_A, features, labels, model_seed):
    dur = []
    
    if args.dataset in ['twitch-gamer', 'Penn94', 'genius', 'tolokers', 'minesweeper', 'roman-empire']: 
        # encouraged to use fixed splits
        data.load_mask()
    else:
        data.load_mask(p=(0.6,0.2,0.2))

    logger.info('#Train:{}'.format(data.train_mask.sum().item()))
    reset_random_seeds(model_seed)
    
    if args.dataset in ['genius', 'minesweeper', 'tolokers', 'questions']:    
        loss_fcn = bce_with_logits_loss
        evaluator = fast_auc_th
    else:
        loss_fcn = F.cross_entropy # input: logits (N, C) and labels (N,)
        evaluator = acc

    data.in_feats = features.shape[-1]

    model = build_model(args, edge_index, norm_A, data.in_feats, data.n_classes)
    optimizers = build_optimizers(args, model)
    stopper = EarlyStopping(patience=args.patience, store_path=args.es_ckpt+'.pt', gauge='acc')
    stopper_step = stopper.step
    
    rec_val_loss = []
    rec_val_accs = []

    for epoch in range(args.n_epochs):
        t0 = time.time()
        
        model.train()
        for _ in optimizers:
            _.zero_grad()
        logits = model(features)
        loss_train = loss_fcn(logits[data.train_mask], labels[data.train_mask])
        loss_train.backward()

        for _ in optimizers:
            _.step()
        
        logits = logits.detach()
        loss_val = loss_fcn(logits[data.val_mask], labels[data.val_mask])
        
        acc_train = evaluate(logits, labels, data.train_mask, evaluator)
        acc_val = evaluate(logits, labels, data.val_mask, evaluator)
        rec_val_loss.append(loss_val.item())
        rec_val_accs.append(acc_val)
        dur.append(time.time() - t0)
        if (epoch+1) % 50 == 0 :
            logger.warning("Epoch {:05d} | Time(s) {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} |  Train Acc {:.4f} | " 
                        "ETputs(KTEPS) {:.2f}".format(epoch+1, np.mean(dur), loss_val.item(),
                                                        acc_val, acc_train, 
                                                        data.n_edges/ np.mean(dur) / 100)
                        )
        if args.early_stop and epoch >= 0:
            if stopper_step(acc_val, model):
                # import ipdb; ipdb.set_trace()
                break   
    # end for

    if args.early_stop:
        model.load_state_dict(th.load(stopper.store_path))
        logger.debug('Model Saved by Early Stopper is Loaded!')
    
    model.eval()
    
    logits = model(features)
    loss_val = loss_fcn(logits[data.val_mask], labels[data.val_mask])
    loss_test = loss_fcn(logits[data.test_mask], labels[data.test_mask])
    acc_val = evaluate(logits, labels, data.val_mask, evaluator)
    acc_test = evaluate(logits, labels, data.test_mask, evaluator)
    logger.info("[FINAL MODEL] Run {} .\Val accuracy {:.2%} \Val loss: {:.2}".format(cv_id+args.start_cv, acc_val, loss_val))
    logger.info("[FINAL MODEL] Run {} .\tTest accuracy {:.2%} \Test loss: {:.2}".format(cv_id+args.start_cv, acc_test, loss_test))

    return model, acc_val, acc_test
    

def main(args):
    reset_random_seeds(args.seed)
    data  = build_dataset(args)
    # Set random split seeds for args.n_cv run
    data.seeds = [random.randint(0,10000) for _ in range(args.n_cv)]
    # Set random model seeds for args.n_cv runs 
    model_seeds = [random.randint(0,10000) for _ in range(args.n_cv)]
    
    logger.info('Split_seeds:{:s}'.format(str(data.seeds)))
    logger.info('Model_seeds:{:s}'.format(str(model_seeds)))

    edge_index = data.edge_index
    if args.graph_norm == 'sym':
        # Alway set `add_self_loops=False' here. 
        # If args.self_loop is True, the self-loops would be loaded in the loader 
        _, norm_A = gcn_norm(edge_index, add_self_loops=False)
    elif args.graph_norm == 'none':
        norm_A = th.ones(data.n_edges, device=data.device)
    else:
        raise NotImplementedError("Case for Rescaled Laplacian Not Implemented!")
        exit(-1)
    
    features = data.features
    labels = data.labels

    accs = []
    val_accs = []
    
    for cv_id in range(args.n_cv):
        model, val_acc, test_acc = run(args, cv_id, edge_index, data, norm_A,  features, labels, model_seed=model_seeds[cv_id])
        test_acc = test_acc.detach().cpu().numpy()
        val_acc = val_acc.detach().cpu().numpy()
        accs.append(test_acc)
        val_accs.append(val_acc)

    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(np.array(accs),func=np.mean,n_boot=1000),95)-np.array(accs).mean()))
    logger.info("Mean Acc For Cross Validation: {:.4f}, STDV: {:.4f}".format(np.array(accs).mean(), np.array(accs).std()))
    logger.info("Uncertainty: {:.4f}".format(uncertainty))
    logger.info(accs)

    
def set_args():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--model", type=str, default='OptBasisGNN',help='(OptBasisGNN, ARMA, BernNet)')
    parser.add_argument("--gpu", type=int, default=1, help="gpu")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--ds-split", type=str, default="standard", help="split by ('standard', 'random').")
    
    # For graph
    parser.add_argument("--self-loop", action='store_true', default=False, help="graph self-loop (default=False)")
    parser.add_argument("--udgraph", action='store_true', default=False, help="process the graph to be undirected (default=False)")
    parser.add_argument("--lcc", action='store_true', default=False)
    parser.add_argument("--graph-norm", type=str, default="none", 
                        choices=['sym', 'none', 'rescale'], 
                        required=True
                        )

    # For model structure configuration 
    parser.add_argument("--n-layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--n-hidden", type=int, default=64, help="number of hidden units")

    # For model training
    parser.add_argument("--wd1", type=float, default=1e-2, help="Weight for L2 loss")
    parser.add_argument("--wd2", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--wd3", type=float, default=5e-4, help="Weight for L2 loss. Used in FavardGNN")
    parser.add_argument("--lr1",  type=float, default=1e-2, help="learning rate")
    parser.add_argument("--lr2",  type=float, default=1e-2, help="learning rate")
    parser.add_argument("--lr3",  type=float, default=1e-2, help="learning rate. Used in FavardGNN")
    parser.add_argument("--alpha",  type=float, default=0.5, help="Option for initialization. Used in GPRGNN")
    parser.add_argument("--momentum",  type=float, default=0.9, help="SGD momentum. Used in ClenshawGCN")
    parser.add_argument("--n-epochs", type=int, default=2000, help="number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--dropout2", type=float, default=0.7, help="dropout probability")

    # For experiment running
    parser.add_argument("--early-stop", action='store_true', default=False, help="early stop (default=False)")
    parser.add_argument("--patience", type=int, default=300, help="patience for early stop")
    parser.add_argument("--es-ckpt", type=str, default="es_checkpoint", help="Saving directory for early stop checkpoint")
    parser.add_argument("--n-cv", type=int, default=1, help="number of cross validation")
    parser.add_argument("--start-cv", type=int, default=0, help="option used in debugging mode")
    # # - For logging
    parser.add_argument("--file-logging", action='store_true', default=False, help="Enable logging to files (default: False)")
    parser.add_argument("--file-log-id", type=int, default=0, help="ID for log directory (default: 0)")
    parser.add_argument("--detailed-console-logging", action='store_true', default=False, help="Enable detailed logging in the console (default: False)")


    args = parser.parse_args()

    if args.gpu < 0:
        args.gpu = 'cpu'

    if args.es_ckpt == 'es_checkpoint':
        args.es_ckpt = '_'.join([args.es_ckpt, 'device='+str(args.gpu)])

    return args


if __name__=='__main__':
    args = set_args()
    logger = _set_logger(args)
    logger.info(args)
    main(args)