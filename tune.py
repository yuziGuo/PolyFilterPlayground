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
from utils.model_utils import build_model, build_optimizers
from utils.stopper import EarlyStopping
from utils.rocauc_eval import eval_rocauc
from utils.model_utils import bce_with_logits_loss
from utils.rocauc_eval import fast_auc_th, fast_auc, acc


def evaluate(logits, labels, mask, evaluator):    
    if not th.is_tensor(logits):
        logits = logits[0]
    logits = logits[mask]
    labels = labels[mask]
    metric = evaluator(logits, labels)
    return metric


def run(args, logger, trial, 
        edge_index, data, norm_A, features, labels, 
        model_seed
        ):
    dur = []


    # split dataset for this run
    if args.dataset in ['twitch-gamer', 'Penn94', 'genius']: 
        # encouraged to use fixed splits
        data.load_mask()
    else:
        # Use random splits
        data.load_mask(p=(0.6,0.2,0.2))
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
    if args.early_stop:
        stopper = EarlyStopping(patience=args.patience, store_path=args.es_ckpt+'.pt')
        stopper_step = stopper.step
    
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
        
        acc_val = evaluate(logits, labels, data.val_mask, evaluator)
        trial.report(acc_val, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        acc_train = evaluate(logits, labels, data.train_mask, evaluator)

        dur.append(time.time() - t0)
        if args.log_detail and (epoch+1) % 20 == 0 :
            logger.info("Epoch {:05d} | Time(s) {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} |  Train Acc {:.4f} | " 
                        "ETputs(KTEPS) {:.2f}".format(epoch+1, np.mean(dur), loss_val.item(),
                                                        acc_val, acc_train, 
                                                        data.n_edges/ np.mean(dur) / 100)
                        )
        if args.early_stop and epoch >= 0:
            if stopper_step(acc_val, model):
                break    
    # end training

    if args.early_stop:
        model.load_state_dict(th.load(stopper.store_path))
        logger.debug('Model Saved by Early Stopper is Loaded!')
    model.eval()
    with th.no_grad():
        logits = model(features)
    loss_val = loss_fcn(logits[data.val_mask], labels[data.val_mask])
    loss_test = loss_fcn(logits[data.test_mask], labels[data.test_mask])
    acc_val = evaluate(logits, labels, data.val_mask, evaluator)
    acc_test = evaluate(logits, labels, data.test_mask, evaluator)
    logger.info("[FINAL MODEL] Val accuracy {:.2%} \Val loss: {:.2}".format(acc_val, loss_val))
    logger.info("[FINAL MODEL] Test accuracy {:.2%} \Test loss: {:.2}".format(acc_test, loss_test))
    return acc_val, acc_test


def main(args, logger, trial):
    reset_random_seeds(args.seed)
    data  = build_dataset(args)
    data.set_split_seeds()
    model_seeds = [random.randint(0,10000)]
    logger.info('Split_seeds:{:s}'.format(str(data.seeds)))
    logger.info('Model_seeds:{:s}'.format(str(model_seeds)))

    edge_index = data.edge_index
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    _, norm_A = gcn_norm(edge_index, add_self_loops=False)
    features = data.features
    labels = data.labels
    
    # I tune hyper-parameter on only the first cv
    cv_id = 0
    val_acc, test_acc = run(args, logger, trial,  
                            edge_index, data, norm_A, features, labels, 
                            model_seed=model_seeds[cv_id]
                            )

    logger.info("Acc on the first split (Validation Set): {:.4f}".format(val_acc))
    logger.info("Acc on the first split (Test Set): {:.4f}".format(test_acc))
    return val_acc.item(), test_acc.item()


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
    parser.add_argument("--n-epochs", type=int, default=2000)

    static_args = parser.parse_args()
    if static_args.gpu < 0:
        static_args.gpu = 'cpu'
    
    ## 1.2 Static options shared by all tasks (Part II)
    dargs = vars(static_args)
    ## 1.3 Static options of the specific model
    dargs.update(public_static_opts)
    if f'{static_args.model}_static_opts' in globals().keys():
        k = f'{static_args.model}_static_opts'
        dargs.update(globals()[k])

    # 2. Args to be tuned
    # Other options to be tuned will be suggested by optuna.
    # For such case, we initialize a `suggestor' here, which wraps functions provided by optuna like `trial.suggest_float'.
    # The suggestor suggests a group of option in a specific run (See function `objective').  
    # Most of the options are shared across different models, i.e., learning rates, weight decays.
    to_tune = public_hypers_default
    if f'{static_args.model}_opts' in globals().keys():
        k = f'{static_args.model}_opts'
        to_tune.update(globals()[k])
    suggestor = convert_dict_to_optuna_suggested(to_tune, static_args.model)
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
    val_acc, test_acc = main(args, logger, trial)
    trial.set_user_attr("val_acc", val_acc)
    trial.set_user_attr("test_acc", test_acc)
    return val_acc


if __name__ == '__main__':
    global static_args
    global suggestor
    static_args, suggestor = initialize_args()

    # create an optuna study
    dataset = static_args.dataset
    kw = f'{static_args.model}-{dataset}'
    study = optuna.create_study(
        study_name="woselfloop-{}".format(dataset),
        direction="maximize", 
        storage = optuna.storages.RDBStorage(url='sqlite:///{}/{}.db'.format('cache/OptunaTrials', kw), 
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
    print("Study statistics this: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", num_pruned)
    print("  Number of complete trials: ", num_completed)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("  Yuhe's record: ")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

    # from utils.optuna_utils import _gen_scripts
    # cmd_str = _gen_scripts(study, vars(static_args))
    # print(cmd_str)