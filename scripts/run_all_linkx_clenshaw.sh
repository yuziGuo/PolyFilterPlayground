# [genius]
# python train_clenshaw_linkx.py --dataset genius --udgraph --self-loop --lr1 0.005 --lr2 0.01 --lr3 0.04 --wd1 1e-6 --wd2 1e-4 --wd3 1e-3 --dropout 0.1 --dropout2 0. --lamda 1.5 --n-layers 8 --momentum 0.8  --early-stop  --log-detail --log-detailedCh 

# [Penn94]
python train_clenshaw_linkx.py --dataset Penn94 --udgraph --self-loop --lr1 0.01 --lr2 0.02 --lr3 0.05 --wd1 1e-8 --wd2 1e-5 --wd3 1e-7 --dropout 0.4 --dropout2 0.1 --lamda 1.0 --n-layers 16 --momentum 0.95   --early-stop  --log-detail --log-detailedCh 

# [gamer]
python train_clenshaw_linkx.py --dataset twitch-gamer --udgraph --self-loop --lr1 0.01 --lr2 0.01 --lr3 0.04 --wd1 1e-7 --wd2 1e-5 --wd3 1e-6 --dropout 0.1 --dropout2 0.1 --lamda 2.0 --n-layers 16 --momentum 0.95   --early-stop    --log-detail --log-detailedCh  