import argparse

import optuna
from optuna.trial import TrialState
import optuna.study

from utils.grading_logger import _set_logger

import ipdb
import numpy as np

from opts.tune.public_hypers import public_hypers_default
from opts.tune.public_hypers import convert_dict_to_optuna_suggested

from utils.optuna_utils import _ckpt_fname
from utils.optuna_utils import _get_complete_and_pruned_trial_nums
from utils.optuna_utils import _pruneDuplicate, _CkptsAndHandlersClearerCallBack



def main():
    return np.random.rand()


def initialize_args():
    # 1. Set static args
    ## 1.1 static options shared by all tasks
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--model", type=str, default='NormalNN')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="cora")
    ## log options
    parser.add_argument("--logging", action='store_true', default=False)
    parser.add_argument("--log-detail", action='store_true', default=False)
    parser.add_argument("--log-detailedCh", action='store_true', default=False)
    parser.add_argument("--id-log", type=int, default=0)
    ##
    parser.add_argument("--optuna-n-trials", type=int, default=202)
    static_args = parser.parse_args()
    if static_args.gpu < 0:
        static_args.gpu = 'cpu'
    
    ## 1.2 Static options shared by all tasks (Part II)
    dargs = vars(static_args)
    from opts.tune.public_static_settings import public_static_opts
    dargs.update(public_static_opts)

    # 2. Args to be tuned
    # Other options to be tuned will be suggested by optuna.
    # For such case, we initialize a `suggestor' here, which wraps functions provided by optuna like `trial.suggest_float'.
    # The suggestor suggests a group of option in a specific run (See function `objective').  
    # Most of the options are shared across different models, i.e., learning rates, weight decays.
    suggestor = convert_dict_to_optuna_suggested(public_hypers_default)    
    return static_args, suggestor


def objective(trial):
    # arguments
    suggested_args = suggestor(trial)
    # args = {} # create an empty namespace object
    args = argparse.Namespace()
    dargs = vars(args)
    dargs.update(vars(static_args))
    dargs.update(suggested_args)
    dargs.update({'es_ckpt': _ckpt_fname(trial.study, trial)})

    # logger
    logger = _set_logger(args)
    logger.info(args)

    # might prune; in this case an exception will be raised
    _pruneDuplicate(trial)

    # report args
    # run
    val_acc = main()
    trial.set_user_attr("val_acc", val_acc)
    
    return val_acc


if __name__ == '__main__':
    global static_args
    global suggestor
    static_args, suggestor = initialize_args()

    # create an optuna study
    dataset = static_args.dataset
    kw = f'noBN-{static_args.dataset}'
    study = optuna.create_study(
        study_name="GATClenRes-{}".format(dataset),
        direction="maximize", 
        storage = optuna.storages.RDBStorage(url='sqlite:///{}/GATClenRes-{}.db'.format('cache/OptunaTrials', kw), 
                engine_kwargs={"connect_args": {"timeout": 10000}}),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=15,interval_steps=1,n_min_trials=5),
        load_if_exists=True
    )
    study.set_system_attr('kw', kw)

    
    # run trials
    n_trials = static_args.optuna_n_trials
    num_completed, num_pruned = _get_complete_and_pruned_trial_nums(study)
    while num_completed + num_pruned < n_trials:
        print('{} trials to go!'.format(n_trials - num_completed - num_pruned))
        # One trial each time
        study.optimize(objective, 
            n_trials=1,
            catch=(RuntimeError,), 
            callbacks=(_CkptsAndHandlersClearerCallBack(),) 
            )
        num_completed, num_pruned = _get_complete_and_pruned_trial_nums(study)
        if num_pruned > 1000:
            break
    
    # report results