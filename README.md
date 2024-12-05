
## Table of contents
1. [Introduction](#i-introduction)
2. [A Quick Demo](#ii-a-quick-demo)
    1. [Step 1: Tuning](#step-1-tune)
    2. [Step 2: Reporting selected params](#step-2-report-the-selected-parameters)
    3. [Step 3: Final Training](#step-3-train-and-get-the-final-result)


## I. Introduction
Hi, `PolyFilterPlayground` is a repo for quick hyperparameter tuning, training and testing for polynomial-filter-based GNNs. It is written by Ph.D. student Yuhe Guo from Group Alg@RUC. I've summarized this repo from my own research experience, and it greatly reduces the amount of work I have to do to try out a model.


This repo is still under construction. Some features of this repo are:

- Support **hyper-parameter tuning** via `Optuna` efficiently. Features of Optuna like `Pruners` are used.
- The records are well-cached. 
- Support most datasets (though some of them use interfaces from `PyG` directly).
- Some modules, such as `ROC-AUC` is much faster than the common practice of using implementations from `scikit-learn`.

**Expectations**. I hope this repo can help you to be productive to try out your ideas. I also hope that it can help you to check the performances of former models with your own practice (The number of models I put under the folder `\model` is small. It would not take much time to add more models if you want to try out. You are also welcomed to **join me** and contribute more models). 

I write a demo to go through this repo. 
For more information, feel free to contact me via guoyuhe@ruc.edu.cn.

## II. A Quick Demo.

### Step 1: Tune.

Use the following command to tune `OptBasisGNN` (I also call it `OptBasisGNN` sometimes) on the `Chameleon` dataset.
To get started quickly, we only test `10` optuna trials, and `50` epochs for each trial. (In real experiments, I suggest using `100` or `200` trials, and `2000` epochs for each trial.)

<!-- ```bash
python tune.py --model OptBasisGNN --dataset geom-chameleon  --gpu 0 --logging --log-detailedCh --optuna-n-trials 10 --n-epochs 50 --study-kw quicktest
``` -->

<!-- ```bash
python tune.py --model OptBasisGNN --dataset geom-chameleon  --gpu 0 --logging --log-detailed-console --log-ignore-steps --optuna-n-trials 10 --n-epochs 50 --study-kw quicktest
``` -->


```bash
python tune.py \
    --model OptBasisGNN \
    --dataset geom-chameleon \
    --gpu 0 \
    --file-logging \
    --file-log-id 1201017801 \
    --detailed-console-logging \
    --optuna-n-trials 10 \
    --n-epochs 50 \
    --study-kw quicktest
```

**Notes for this step.**

1. **Optuna cache**. This command creates a directory in `cache/OptunaTrials/`. This is where `Optuna` gathers the trial histories and uses them to search for the next hyper-parameters. 
The caches are organized with study keywords, branch id, and commit id (for different versions of the same model). 
The zipped options are also stored in the same folder. The structure is as follows:

    ```
    cache/
    ├── ckpts/
    └── OptunaTrials/
        └──studykw=[studykw]||br=[branchid]||cmt=[cmtid]/   # Ensure the same model is used in the same Optuna .db file. When the model or data undergoes slight changes, a new .db file will be created in a separate folder to avoid confusion.
            ├── zipped-opts|[model]|[dataset]|[run_time]/   # Zip the files that decide the static and dynamic options, e.g., in case that we forget whether the runing trials adds self-loops or not. 
            └── OptBasisGNN-geom-chameleon.db               # Optuna database.
    ```

2. **Logging Modes**. Here, `--file-logging --file-log-id --detailed-console-logging` are all log options. They mean that detailed logs are printed on the console and also saved in the files under `logs/[arg.file-log-id]/`.

You can also specify other logging forms. For example, `--file logging --file-log-id [xxx]` will make it **more silent** in the console, and instead, log the details for each trial into the folder `logs/[arg.file-log-id]/`. I put some examples in `cmds_tune.sh`:

    ```bash
    name="opt-roman-empire"
    python tune.py --study-kw quicktest \
        --model OptBasisGNN --dataset roman-empire --gpu 0 \
        --file-logging   --file-log-id  1201017801 \
        1>runs/logs/${name}.log  2>runs/logs/${name}.err & 
    ```

3. **How is the search space of hyperparameters specified?** I put them in `opts/`. Please read the logic in `initialize_args()` in `tune.py`. If you want to tune your own model, remember to specify some options under `opts/` as well. (Also remember to slightly trim `build_optimizers()` and `build_models()` for your own practice.)

### Step 2: Report the selected parameters.

Following Step 1, now let the selected hyper-parameters be reported:

    ```bash
    python report.py --study-kw quicktest --model OptBasisGNN --dataset geom-chameleon  --gpu 0 --file-logging  --detailed-console-logging --optuna-n-trials 10 
    ```

The output is as follows. You can directly copy it to train!
    
    ```bash
    python train.py --dropout 0.5 --dropout2 0.5 --lr1 0.05 --lr2 0.03 --n-layers 10 --wd1 1e-07 --wd2 1e-05 --seed 42 --model OptBasisGNN --dataset geom-chameleon --logging   --log-detailedCh   --udgraph   --early-stop   --n-hidden 64 --n-cv 10 --es-ckpt OptBasisGNN-geom-chameleon --log-detail --log-detailedCh 1>logs/OptBasisGNN-geom-chameleon.log 2>logs/OptBasisGNN-geom-chameleon.err&
    ```

**Remarks**:
1. The command above seem a bit verbose. Actually, most parameters like `--early-stop`, `--logging` and `--udgraph`(using undirected graph, which is a common practice in polynomial-filter based GNNs) are the same among all the practices. 

1. Note that if the number of `--optuna-n-trials` you assign is **larger** than the *finished trials* in the tuning process, 
the result will not be reported.

2. Make sure the name of `--model` and `--dataset` is the same in the Step 1 and Step 2. They decides the name of Optuna cache database that are you using.

### Step 3: Train! And get the final result.
Just copy the output in Step 2 to run! 
Since we specified `1>logs/OptBasisGNN-geom-chameleon.log 2>logs/OptBasisGNN-geom-chameleon.err&`, 
the results can be found in `logs/OptBasisGNN-geom-chameleon.log`.

![Alt text](image.png)

We find that even tuned for such a small number of `10` optuna trials, and `50` epochs for each trial, the final result, `73.98±0.88` over 10 random splits is quite high.