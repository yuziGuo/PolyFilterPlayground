import numpy as np
from sklearn.metrics import roc_auc_score
import torchmetrics
import torch.nn.functional as F
import torch as th
from typing import Union


# yts = th.tensor([210979]).to('cuda:1')


def acc(logits, labels):
    _, indices = th.max(logits, dim=1)
    correct = th.sum(indices == labels)
    acc = correct.item() * 1.0 / len(labels)
    return acc


def fast_auc_th(
        y_score: np.array,
        y_true: np.array,  
        sample_weight: np.array=None) -> Union[float, str]:

    yts = th.tensor(y_true.shape, device=y_true.device) - 1
    zero = th.tensor([0], device=y_true.device)

    # binary clf curve
    y_score = F.softmax(y_score, dim=-1)[:,1]
    y_true = (y_true == 1)
    desc_score_indices = th.argsort(y_score).flip(dims=[0])
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        sample_weight = sample_weight[desc_score_indices]

    distinct_value_indices = th.where(th.diff(y_score))[0]
    threshold_idxs = th.concat((distinct_value_indices, yts))

    if sample_weight is not None:
        tps = th.cumsum(y_true * sample_weight, dim=0)[threshold_idxs]
        fps = th.cumsum((1 - y_true) * sample_weight,dim=0)[threshold_idxs]
    else:
        tps = th.cumsum(y_true.int(),dim=0)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

    # roc
    tps = th.concat((zero, tps))
    fps = th.concat((zero, fps))

    if fps[-1] <= 0 or tps[-1] <= 0:
        return th.nan
    
    direction = 1
    dx = th.diff(fps)
    if th.any(dx < 0):
        direction = -1

    area = direction * th.trapz(tps, fps) / (tps[-1] * fps[-1])
    return area


def fast_auc_th_(y_true: np.array, 
             y_score: np.array, 
             sample_weight: np.array=None) -> Union[float, str]:

    # binary clf curve
    y_score = F.softmax(y_score, dim=-1)[:,1]
    # y_true = (y_true == 1)
    desc_score_indices = th.argsort(y_score).flip(dims=[0])
    # import ipdb; ipdb.set_trace()
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        sample_weight = sample_weight[desc_score_indices]

    distinct_value_indices = th.where(th.diff(y_score))[0]
    # import ipdb; ipdb.set_trace()
    # threshold_idxs = np.r_[c, y_true.size - 1]
    # threshold_idxs = th.concat((distinct_value_indices, th.tensor(y_true.shape).to(y_true.device) - 1))
    yts = th.tensor(y_true.shape, device=y_true.device) - 1
    threshold_idxs = th.concat((distinct_value_indices, yts))

    if sample_weight is not None:
        tps = th.cumsum(y_true * sample_weight, dim=0)[threshold_idxs]
        fps = th.cumsum((1 - y_true) * sample_weight,dim=0)[threshold_idxs]
    else:
        tps = th.cumsum(y_true.int(),dim=0)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

    # roc
    tps = th.concat((zero, tps))
    fps = th.concat((zero, fps))

    if fps[-1] <= 0 or tps[-1] <= 0:
        return th.nan

    # auc
    direction = 1
    dx = th.diff(fps)
    if th.any(dx < 0):
        if th.all(dx <= 0):
            direction = -1
        else:
            return 'error'

    area = direction * th.trapz(tps, fps) / (tps[-1] * fps[-1])
    return area


def fast_auc(y_true: np.array, 
             y_score: np.array, 
             sample_weight: np.array=None) -> Union[float, str]:
    """a function to calculate AUC via python.

    Args:
        y_true (np.array): 1D numpy array as true labels.
        y_score (np.array): 1D numpy array as probability predictions.
        sample_weight (np.array): 1D numpy array as sample weights, optional.

    Returns:
        float or str: AUC score or 'error' if imposiible to calculate
    """
    # binary clf curve
    y_score = F.softmax(y_score, dim=-1)[:,1].detach().cpu().numpy()
    y_true = (y_true == 1).detach().cpu().numpy()
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    import ipdb; ipdb.set_trace()
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        sample_weight = sample_weight[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    if sample_weight is not None:
        tps = np.cumsum(y_true * sample_weight)[threshold_idxs]
        fps = np.cumsum((1 - y_true) * sample_weight)[threshold_idxs]
    else:
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

    # roc
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan

    # auc
    direction = 1
    dx = np.diff(fps)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            return 'error'

    area = direction * np.trapz(tps, fps) / (tps[-1] * fps[-1])
    return area


def eval_rocauc(y_true, y_pred):
    y_true_arr = y_true.detach().cpu().numpy()
    y_pred = F.softmax(y_pred, dim=-1)[:,1]
    y_pred_arr = y_pred.detach().cpu().numpy()
    v1 = roc_auc_score(y_true_arr, y_pred_arr)
    # v3 = torchmetrics.AUROC(task='binary')(y_pred, y_true).item()
    return v1

def _eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)
    y_true = y_true.detach().cpu().numpy()

    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()  # why

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])   
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')
    import ipdb; ipdb.set_trace()
    return sum(rocauc_list)/len(rocauc_list)