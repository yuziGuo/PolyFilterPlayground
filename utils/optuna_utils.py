from optuna.trial import TrialState

import os
import numpy as np
import logging
import optuna


def _ckpt_fname(study, trial):
    return  '_'.join([study.study_name, study.system_attrs['kw'],'trialNo{}'.format(trial.number)])


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


# def _gen_scripts(study, static_args):
#     trial = study.best_trial
#     cmd_opt_strs = []
#     for k, v in trial.params.items():
#         _opt = f"--{k} {v}"
#         cmd_opt_strs.append(_opt)
#     for k,v in static_args.items():
#         _opt = f"--{k} {v}"
#         cmd_opt_strs.append(_opt)
#     cmd_str = ' '.join(cmd_opt_strs)
#     return cmd_str