# # [film]
python train_clenshaw.py  --dataset geom-film --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.0 --dropout2 0.0 --lamda 2.0 --lr1 0.04 --lr2 0.05 --lr3 0.03 --momentum 0.95 --n-layers 20 --wd1 1e-8 --wd2 1e-3 --wd3 1e-5 

# # [squirrel]
python train_clenshaw.py  --dataset geom-squirrel --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.5 --dropout2 0.1 --lamda 1.0 --lr1 0.01 --lr2 0.04 --lr3 0.05 --momentum 0.95 --n-layers 8 --wd1 1e-4 --wd2 1e-5 --wd3 1e-8 

# [chameleon]
python train_clenshaw.py  --dataset geom-chameleon --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.6 --dropout2 0.0 --lamda 1.0 --lr1 0.01 --lr2 0.02 --lr3 0.01 --momentum 0.95 --n-layers 8 --wd1 1e-5 --wd2 1e-3 --wd3 1e-7 

# [corafull]
python train_clenshaw.py  --dataset corafull --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.7 --dropout2 0.0 --lamda 1.5 --lr1 0.01 --lr2 0.04 --lr3 0.03 --momentum 0.8 --n-layers 8 --wd1 1e-8 --wd2 1e-3 --wd3 1e-4 

# [pubmedfull]
python train_clenshaw.py  --dataset pubmedfull --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.1 --dropout2 0.1 --lamda 1.0 --lr1 0.02 --lr2 0.03 --lr3 0.05 --momentum 0.85 --n-layers 16 --wd1 1e-4 --wd2 1e-3 --wd3 1e-7 

# [citeseerfull]
python train_clenshaw.py  --dataset citeseerfull --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.7 --dropout2 0.2 --lamda 1.5 --lr1 0.05 --lr2 0.005 --lr3 0.005 --momentum 0.8 --n-layers 28 --wd1 1e-6 --wd2 1e-7 --wd3 1e-5 

# [cornell]
python train_clenshaw.py  --dataset geom-cornell --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.2 --dropout2 0.5 --lamda 2.0 --lr1 0.01 --lr2 0.05 --lr3 0.02 --momentum 0.8 --n-layers 16 --wd1 1e-3 --wd2 1e-4 --wd3 1e-3 

# [texas]
python train_clenshaw.py  --dataset geom-texas --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.2 --dropout2 0.6 --lamda 1.0 --lr1 0.001 --lr2 0.05 --lr3 0.05 --momentum 0.85 --n-layers 8 --wd1 1e-3 --wd2 1e-3 --wd3 1e-3

# [photo]
python train_clenshaw.py  --dataset photofull --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.3 --dropout2 0.5 --lamda 1.0 --lr1 0.05 --lr2 0.001 --lr3 0.005 --momentum 0.9 --n-layers 16 --wd1 1e-6 --wd2 1e-8 --wd3 1e-5

python train_clenshaw.py  --dataset photofull --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.4 --dropout2 0.0 --lamda 1.5 --lr1 0.02 --lr2 0.005 --lr3 0.01 --momentum 0.8 --n-layers 8 --wd1 1e-5 --wd2 1e-3 --wd3 1e-8
# [32m[I 2023-04-09 10:06:01,569][0m Trial 18 finished with value: 0.9562091503267974 and parameters: {'lr1': 0.02, 'lr2': 0.030000000000000002, 'lr3': 0.02, 'wd1': -5, 'wd2': -4, 'wd3': -4, 'dropout': 0.4, 'dropout2': 0.5, 'lamda': 2.0, 'n_layers': 4, 'momentum': 0.9}. Best is trial 18 with value: 0.9562091503267974.[0m

#[computers]
python train_clenshaw.py  --dataset computersfull --early-stop  --udgraph --self-loop         --loss nll  --n-cv 20 --log-detail --log-detailedCh --dropout 0.5 --dropout2 0.2 --lamda 1.5 --lr1 0.01 --lr2 0.01 --lr3 0.04 --momentum 0.8 --n-layers 16 --wd1 1e-6 --wd2 1e-4 --wd3 1e-4
# [32m[I 2023-04-09 09:46:59,524][0m Trial 23 finished with value: 0.9254545454545454 and parameters: {'lr1': 0.009999999999999998, 'lr2': 0.009999999999999998, 'lr3': 0.039999999999999994, 'wd1': -6, 'wd2': -4, 'wd3': -4, 'dropout': 0.5, 'dropout2': 0.2, 'lamda': 1.5, 'n_layers': 16, 'momentum': 0.8}. Best is trial 23 with value: 0.9254545454545454.[0m
