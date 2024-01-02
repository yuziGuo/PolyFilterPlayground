from tune import *

import argparse
import optuna
import optuna.study

import numpy as np
import random 
import torch as th
import torch.nn.functional as F

import time

from opts.tune.public_hypers import public_hypers_default
from opts.tune.public_hypers import convert_dict_to_optuna_suggested
from opts.tune.public_static_settings import public_static_opts
from opts.tune.public_hypers import public_hypers_default
from opts.tune.private_static_settings import *
from opts.tune.private_hypers import *

from utils.optuna_utils import _ckpt_fname
from utils.optuna_utils import _get_complete_and_pruned_trial_nums
from utils.optuna_utils import _pruneDuplicate, _CkptsAndHandlersClearerCallBack
from utils.random_utils import reset_random_seeds
from utils.data_utils import build_dataset
from utils.grading_logger import _set_logger
from utils.model_utils import build_model, build_optimizers, build_model_augmented
from utils.stopper import EarlyStopping
from utils.rocauc_eval import eval_rocauc
from utils.model_utils import bce_with_logits_loss
from utils.rocauc_eval import fast_auc_th, fast_auc, acc

from torch_geometric.nn.conv.gcn_conv import gcn_norm


def parse_args():
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
    parser.add_argument("--optuna-n-trials", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=2000)

    parser.add_argument("--random-perturb", action='store_true', default=False)
    

    static_args = parser.parse_args()
    if static_args.gpu < 0:
        static_args.gpu = 'cpu'
    return static_args
    
def initialize_static_args(model, dataset):
    # 1. Set static args
    ## 1.1 static options shared by all tasks
    static_args = parse_args()
    
    ## 1.2 Static options shared by all tasks (Part II)
    dargs = vars(static_args)
    dargs.update({"model": model, "dataset": dataset})

    ## 1.3 Static options of the specific model
    dargs.update(public_static_opts)
    if f'{static_args.model}_static_opts' in globals().keys():
        k = f'{static_args.model}_static_opts'
        dargs.update(globals()[k])
    
    return static_args
    


if __name__ == '__main__':
    for model in ['OptBasisGNN', 'FavardGNN']:
        for dataset in ['minesweeper', 'tolokers', 'roman-empire',
                #  'cs', 'physics'
                 ]:
            static_args = initialize_static_args(model, dataset)

            # create an optuna study
            kw = f'{model}-{dataset}'
            study = optuna.create_study(
                study_name="{}".format(dataset),
                direction="maximize", 
                storage = optuna.storages.RDBStorage(url='sqlite:///{}/{}.db'.format('cache/OptunaTrials', kw)),
                load_if_exists=True
            )
            study.set_system_attr('kw', kw)
            
            n_trials = static_args.optuna_n_trials
            num_completed, num_pruned = _get_complete_and_pruned_trial_nums(study)
            if num_completed + num_pruned >= n_trials:
                print("Finished! Now I will report.")
            else:
                print('There remains {} trials to go!'.format(n_trials - num_completed - num_pruned))

            from utils.optuna_utils import _gen_scripts
            tofile = f"--es-ckpt {kw} --log-detail --log-detailedCh 1>logs/{kw}.log 2>logs/{kw}.err&"
            cmd_str = _gen_scripts(study, vars(static_args), prefix="python train.py", postfix=f"--n-cv 10 {tofile}")
            print(cmd_str, file=open('cmds.sh', 'a'))