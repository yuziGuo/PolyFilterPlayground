from optuna.trial import TrialState

import os
import numpy as np
import logging
import optuna

from utils.exp_utils import get_branch_name, get_commit_id

def _ckpt_fname(study, trial):
    return  '||'.join([study.study_name, 
                      f"kw={study.user_attrs['kw']}",
                      f'trialNo{trial.number}',
                      f'br={get_branch_name()}',
                      f'cmt={get_commit_id()[:6]}',
                      ])


def _pruneDuplicate(trial):
    trials = trial.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    numbers=np.array([t.number for t in trials])
    bool_params= np.array([trial.params==t.params for t in trials]).astype(bool)
    #Don´t evaluate function if another with same params has been/is being evaluated before this one
    if np.sum(bool_params)>1:
        if trial.number>np.min(numbers[bool_params]):
            logging.getLogger('optuna.pruners').info('[YH INFO] Prune duplicated args!')
            raise optuna.exceptions.TrialPruned()
    return 

def _get_complete_and_pruned_trial_nums(study):
    num_completed = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
    num_pruned = len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED]))
    return num_completed, num_pruned


class _CkptsAndHandlersClearerCallBack:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __call__(self, study, Frozentrial) -> None:
        # clean logger
        logger=logging.getLogger("detail")
        logger.handlers = []
        # clean ckpt
        for suffix in ['', '-0', '-1', '-2', '-3', '-4']:
            _p = os.path.join('cache','ckpts', _ckpt_fname(study, Frozentrial)  +'.pt' + suffix)
            if os.path.exists(_p):
                os.remove(_p)


def _process_kv(k, v):
    map_lr = {-0.02: 0.0005,  -0.01: 0.001, 0. : 0.005}
    
    filtered_key_lst = ['optuna_n_trials', 'start_cv', 'n_cv', 'gpu', 'id_log',  
                        'n_epochs', 'patience', 'study_kw']
    
    if k in filtered_key_lst:
        return None, None

    if type(v) == bool:
        if v is False:
            return None, None
        else:
            v = ' '
    if len(k.split('_')) > 0:
        k = '-'.join(k.split('_'))
    if k.startswith('lr'):
        if v<=0:
            v = map_lr[v]
        v = round(v, 4)
    if k.startswith('wd'):
        v = float('1e'+str(v))
    return k, v


def _gen_scripts(study, 
                 static_args, 
                 prefix="python train.py", 
                 postfix="--n-cv 20"
                 ):
    trial = study.best_trial
    best_trial_test_acc = round(trial.user_attrs['test_acc'], 4)
    best_trial_val_acc = round(trial.user_attrs['val_acc'], 4)

    cmd_opt_strs = [prefix]
    for k, v in trial.params.items():
        k,v = _process_kv(k,v)
        if k is None:
            continue
        _opt = f"--{k} {v}"
        cmd_opt_strs.append(_opt)
    for k,v in static_args.items():
        k,v = _process_kv(k,v)
        if k is None:
            continue
        _opt = f"--{k} {v}"
        cmd_opt_strs.append(_opt)
    cmd_opt_strs.append(postfix)
    cmd_str = ' '.join(cmd_opt_strs)
    cmd_str = cmd_str + f"\n# Trial acc(test/val): {best_trial_test_acc}/{best_trial_val_acc} \n"

    return cmd_str