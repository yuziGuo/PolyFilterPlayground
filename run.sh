# name="opt-genius"
# python tune.py --model NormalNN --dataset genius --gpu 1 --logging --log-detail --id-log 1011014501 1>>logs/${name}.log  2>>logs/${name}.err &
# sleep 3

# name="opt-cham"
# python tune.py --model NormalNN --dataset geom-chameleon --gpu 0 --logging --log-detail --id-log 1011014502 1>>logs/${name}.log  2>>logs/${name}.err &

# name="gpr-cham"
# python tune.py --model GPRGNN --dataset geom-chameleon --gpu 0 --logging --log-detail --id-log 1011014503 1>>logs/${name}.log  2>>logs/${name}.err &

# name="gpr-squirrel"
# python tune.py --model GPRGNN --dataset geom-squirrel --gpu 1 --logging --log-detail --id-log 1011014504 1>>logs/${name}.log  2>>logs/${name}.err &

# name="optv2-cham"
# python tune.py --model NormalNNV2 --dataset geom-chameleon --gpu 0 --logging --log-detail --id-log 1011014505 1>>logs/${name}.log  2>>logs/${name}.err &

# name="optv2-sq"
# python tune.py --model NormalNNV2 --dataset geom-squirrel --gpu 0 --logging --log-detail --id-log 1011014506 1>>logs/${name}.log  2>>logs/${name}.err &

# name="optv2-wonoise-cham"
# python tune.py --model NormalNNV2 --dataset geom-chameleon --gpu 1 --logging --log-detail --id-log 1012014501 1>>logs/${name}.log  2>>logs/${name}.err &

# name="optv2-wonoise-cham-2"
# python tune.py --model NormalNNV2 --dataset geom-chameleon --gpu 0 --logging --log-detail --id-log 1012014501 1>>logs/${name}.log  2>>logs/${name}.err &

# name="optv2-wonoise-sq"
# python tune.py --model NormalNNV2 --dataset geom-squirrel --gpu 0 --logging --log-detail --id-log 1012014502 1>>logs/${name}.log  2>>logs/${name}.err &

# name="optv2-wonoise-sq-2"
# python tune.py --model NormalNNV2 --dataset geom-squirrel --gpu 0 --logging --log-detail --id-log 1012014502 1>>logs/${name}.log  2>>logs/${name}.err &

# name="gprv2-cham"
# python tune.py --model GPRGNNV2 --dataset geom-chameleon --gpu 0 --logging --log-detail --id-log 1012014503 1>>logs/${name}.log  2>>logs/${name}.err &

# sleep 1

# name="gprv2-cham-2"
# python tune.py --model GPRGNNV2 --dataset geom-chameleon --gpu 0 --logging --log-detail --id-log 1012014503 1>>logs/${name}.log  2>>logs/${name}.err &

# sleep 1

# name="gprv2-sq"
# python tune.py --model GPRGNNV2 --dataset geom-squirrel --gpu 0 --logging --log-detail --id-log 1012014504 1>>logs/${name}.log  2>>logs/${name}.err &

# sleep 1

# name="gprv2-sq-2"
# python tune.py --model GPRGNNV2 --dataset geom-squirrel --gpu 1 --logging --log-detail --id-log 1012014504 1>>logs/${name}.log  2>>logs/${name}.err &

# python train.py  --model NormalNNV2 --dataset geom-chameleon --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0. --dropout2 0.5 --lr1 0.03 --lr2 0.03 --n-layers 12 --wd1 1e-5 --wd2 1e-3  --n-cv 20  --gpu 0 1>cham-optv22.log 2>cham-optv22.err &

#  python train.py  --model NormalNNV2 --dataset geom-squirrel --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0. --dropout2 0.6 --lr1 0.005 --lr2 0.04 --n-layers 4 --wd1 1e-6 --wd2 1e-7  --n-cv 20  --gpu 1 1>sq-optv22.log 2>sq-optv22.err &

# python train.py  --model GPRGNNV2 --dataset geom-chameleon --udgraph  --log-detail --log-detailedCh --early-stop --alpha 0.5 --dropout 0.4 --dropout2 0.7 --lr1 0.04 --lr2 0.05 --n-layers 20 --wd1 1e-4 --wd2 1e-8  --n-cv 20  --gpu 0 1>cham-gprv2.log 2>cham-gprv2.err &

# python train.py  --model GPRGNNV2 --dataset geom-squirrel --udgraph  --log-detail --log-detailedCh --early-stop --alpha 0.6 --dropout 0.5 --dropout2 0.3 --lr1 0.03 --lr2 0.04 --n-layers 12 --wd1 1e-6 --wd2 1e-8  --n-cv 20  --gpu 1 1>sq-gprv2.log 2>sq-gprv2.err &


#######################################

# name="opt-genius-correct"
# python tune.py --model NormalNN --dataset genius --gpu 0 --logging --log-detail --id-log 1013014501 1>>logs/${name}.log  2>>logs/${name}.err &

# name="optv2-noloop-witnoise-cham"
# python tune.py --model NormalNNV2 --dataset geom-chameleon --gpu 1 --logging --log-detail --id-log 1013014502 1>>logs/${name}.log  2>>logs/${name}.err &

# name="optv2-noloop-withnoise-sq"
# python tune.py --model NormalNNV2 --dataset geom-squirrel --gpu 1 --logging --log-detail --id-log 1013014503 1>>logs/${name}.log  2>>logs/${name}.err &


# name="gpr-noloop-cham"
# python tune.py --model GPRGNN --dataset geom-chameleon --gpu 0 --logging --log-detail --id-log 1013014504 1>>logs/${name}.log  2>>logs/${name}.err &

# name="gpr-noloop-squirrel"
# python tune.py --model GPRGNN --dataset geom-squirrel --gpu 1 --logging --log-detail --id-log 1013014505 1>>logs/${name}.log  2>>logs/${name}.err &
# j

# python train.py  --model NormalNNV2 --dataset geom-chameleon --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0. --dropout2 0.4 --lr1 0.05 --lr2 0.02 --n-layers 16 --wd1 1e-7 --wd2 1e-3  --n-cv 20  --gpu 0 1>cham-noloop-optv22.log 2>cham-noloop-optv22.err &

# python train.py  --model NormalNNV2 --dataset geom-squirrel --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0. --dropout2 0.2 --lr1 0.01 --lr2 0.03 --n-layers 12 --wd1 1e-6 --wd2 1e-8  --n-cv 20  --gpu 1 1>sq-noloop-optv22.log 2>sq-noloop-optv22.err &

# name="gprv2-noloop-squirrel"
# python tune.py --model GPRGNNV2 --dataset geom-squirrel --gpu 1 --logging --log-detail --id-log 1014014501 1>>logs/${name}.log  2>>logs/${name}.err &

# name="gprv2-noloop-cham"
# python tune.py --model GPRGNNV2 --dataset geom-chameleon --gpu 1 --logging --log-detail --id-log 1014014502 1>>logs/${name}.log  2>>logs/${name}.err &

# python train.py  --model GPRGNNV2 --dataset geom-chameleon --udgraph  --log-detail --log-detailedCh --early-stop --alpha 0.9 --dropout 0.4 --dropout2 0.7 --lr1 0.05 --lr2 0.05 --n-layers 16 --wd1 1e-6 --wd2 1e-6  --n-cv 20  --gpu 0 1>cham-noloop-gprv2.log 2>cham-noloop-gprv2.err &

# name="optv2-noloop-wonoise-cham"
# python tune.py --model NormalNNV2 --dataset geom-chameleon --gpu 0 --logging --log-detail --id-log 1013014503 1>>logs/${name}.log  2>>logs/${name}.err &

# name="optv2-noloop-wonoise-sq"
# python tune.py --model NormalNNV2 --dataset geom-squirrel --gpu 1 --logging --log-detail --id-log 1013014504 1>>logs/${name}.log  2>>logs/${name}.err &


# python train.py  --model NormalNNV2 --dataset geom-squirrel --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0.4 --dropout2 0.3 --lr1 0.05 --lr2 0.05 --n-layers 16 --wd1 1e-8 --wd2 1e-4  --n-cv 20  --gpu 0 1>sq-wonoise-noloop-optv2.log 2>sq-wonoise-noloop-optv2.err &

# python train.py  --model GPRGNNV2 --dataset geom-squirrel --udgraph  --log-detail --log-detailedCh --early-stop --alpha 0.3 --dropout 0.3 --dropout2 0.8 --lr1 0.05 --lr2 0.05 --n-layers 8 --wd1 1e-5 --wd2 1e-5  --n-cv 20  --gpu 1 1>sq-noloop-gprv2.log 2>sq-noloop-gprv2.err &

# name="opt-genius-correct"
# python tune.py --model NormalNN --dataset genius --gpu 1 --logging --log-detail --id-log 1014014504 1>logs/${name}.log  2>logs/${name}.err &

# name="opt-genius-correct-0"
# python tune.py --model NormalNN --dataset genius --gpu 0 --logging --log-detail --id-log 1015014501 1>logs/${name}.log  2>logs/${name}.err &


# sleep 3
# name="opt-genius-correct-1"
# python tune.py --model NormalNN --dataset genius --gpu 1 --logging --log-detail --id-log 1015014501 1>>logs/${name}.log  2>>logs/${name}.err &

# name="opt-minesweeper"
# python tune.py --model NormalNN --dataset minesweeper --gpu 0 --logging --log-detail --id-log 1015014501 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="optv2-minesweeper"
# python tune.py --model NormalNNV2 --dataset minesweeper --gpu 0 --logging --log-detail --id-log 1015014501 1>logs/${name}.log  2>logs/${name}.err &

# name="opt-toloker"
# python tune.py --model NormalNN --dataset tolokers --gpu 1 --logging --log-detail --id-log 1016014501 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="opt-questions"
# python tune.py --model NormalNN --dataset questions --gpu 1 --logging --log-detail --id-log 1016014502 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3


# name="roman-empire"
# python tune.py --model NormalNN --dataset roman-empire --gpu 0 --logging --log-detail --id-log 1015014501 1>logs/${name}.log  2>logs/${name}.err &

# python train.py  --model NormalNN --dataset tolokers --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0.2 --dropout2 0.4 --lr1 0.04 --lr2 0.03 --n-layers 12 --wd1 1e-7 --wd2 1e-7  --n-cv 10  --gpu 1

# python train.py  --model NormalNN --dataset minesweeper --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0. --dropout2 0. --lr1 0.05 --lr2 0.05 --n-layers 8 --wd1 1e-8 --wd2 1e-3  --n-cv 20  --gpu 1 1>minesweeper-opt.log 2>minesweeper-opt.err &

# name="opt-chamF"
# python tune.py --model NormalNN --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1018014501 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="opt-sqF"
# python tune.py --model NormalNN --dataset squirrelF --gpu 0 --logging --log-detail --id-log 1018014502 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# python train.py  --model NormalNN --dataset squirrelF --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0.8 --dropout2 0.7 --lr1 0.05 --lr2 0.03 --n-layers 4 --wd1 1e-4 --wd2 1e-8  --n-cv 10  --gpu 1 1>sqF-opt.log 2>sqF-opt.err &
# sleep 3

# python train.py  --model NormalNN --dataset chameleonF --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0.3 --dropout2 0.8 --lr1 0.005 --lr2 0.04 --n-layers 20 --wd1 1e-4 --wd2 1e-4  --n-cv 10  --gpu 1 1>chamF-opt.log 2>chamF-opt.err &


# name="gpr-chamF"
# python tune.py --model GPRGNN --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1018014503 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gpr-sqF"
# python tune.py --model GPRGNN --dataset squirrelF --gpu 0 --logging --log-detail --id-log 1018014504 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprv2-chamF"
# python tune.py --model GPRGNNV2 --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1018014503 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprv2-sqF"
# python tune.py --model GPRGNNV2 --dataset squirrelF --gpu 0 --logging --log-detail --id-log 1018014504 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAug-chamF-perturbagain"
# python tune.py --model GPRGNNAugmented --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1026014504 1>logs/${name}.log  2>logs/${name}.err &

# name="gprAug-chamF-perturbagain"
# python tune.py --model GPRGNNAugmented --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1026014504 1>logs/${name}-2.log  2>logs/${name}-2.err &


# name="gprAug-sqF-perturbagain"
# python tune.py --model GPRGNNAugmented --dataset squirrelF --gpu 0 --logging --log-detail --id-log 1026014503 1>logs/${name}.log  2>logs/${name}.err &

# python train.py  --model GPRGNN --dataset squirrelF --udgraph  --log-detail --log-detailedCh --early-stop --alpha 0.3 --dropout 0.7 --dropout2 0. --lr1 0.03 --lr2 0.05 --n-layers 4 --wd1 1e-4 --wd2 1e-5  --n-cv 10  --gpu 1  --es-ckpt e1  1>sqF-gpr.log 2>sqF-gpr.err &
# sleep 3

# python train.py  --model GPRGNN --dataset chameleonF --udgraph  --log-detail --log-detailedCh --early-stop --alpha 0.1 --dropout 0.6 --dropout2 0.3 --lr1 0.03 --lr2 0.001 --n-layers 12 --wd1 1e-3 --wd2 1e-4  --n-cv 10  --gpu 1 --es-ckpt e11  1>chamF-gpr.log 2>chamF-gpr.err &
# sleep 3

# python train.py  --model GPRGNNAugmented --dataset chameleonF --udgraph  --log-detail --log-detailedCh --early-stop --alpha 0.9 --dropout 0.7 --dropout2 0.4 --lr1 0.005 --lr2 0.05 --n-layers 8 --wd1 1e-6 --wd2 1e-7  --n-cv 10  --gpu 1  --es-ckpt e111 1>chamF-gprAug.log 2>chamF-gprAug.err &
# sleep 3

#  python train.py  --model GPRGNNAugmented --dataset squirrelF --udgraph  --log-detail --log-detailedCh --early-stop --alpha 0.2 --dropout 0.9 --dropout2 0. --lr1 0.0005 --lr2 0.02 --n-layers 8 --wd1 1e-3 --wd2 1e-8  --n-cv 10  --gpu 1  --es-ckpt e1111 1>sqF-gprAug.log 2>sqF-gprAug.err &

# name="gprAug-chamF"
# python tune.py --model GPRGNNAugmented --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1026014501 1>logs/${name}-2.log  2>logs/${name}-2.err &
# sleep 3

# name="gprAug-sqF"
# python tune.py --model GPRGNNAugmented --dataset squirrelF --gpu 0 --logging --log-detail --id-log 1026014502 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAug-chamF"
# python tune.py --model GPRGNNAugmented --dataset chameleonF --gpu 1 --logging --log-detail --id-log 1026014501 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAug-sqF"
# python tune.py --model GPRGNNAugmented --dataset squirrelF --gpu 1 --logging --log-detail --id-log 1026014502 1>logs/${name}-2.log  2>logs/${name}-2.err &
# sleep 3

# name="gprAugV2-chamF"
# python tune.py --model GPRGNNV2Augmented --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1026014505 1>logs/${name}-2.log  2>logs/${name}-2.err &
# sleep 3

# name="gprAugV2-sqF"
# python tune.py --model GPRGNNV2Augmented --dataset squirrelF --gpu 0 --logging --log-detail --id-log 1026014506 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAugV2-chamF"
# python tune.py --model GPRGNNV2Augmented --dataset chameleonF --gpu 1 --logging --log-detail --id-log 1026014507 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAugV2-sqF"
# python tune.py --model GPRGNNV2Augmented --dataset squirrelF --gpu 1 --logging --log-detail --id-log 1026014508 1>logs/${name}-2.log  2>logs/${name}-2.err &
# sleep 3

# name="gprAugV2-randP-chamF"
# python tune.py --model GPRGNNV2Augmented --dataset chameleonF --gpu 1 --logging --log-detail --id-log 1026014509 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAugV2-randP-chamF"
# python tune.py --model GPRGNNV2Augmented --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1026014510 1>logs/${name}-1.log  2>logs/${name}-1.err &
# sleep 3

#################################
# name="gprAugV2-P1027-sqF"
# python tune.py --model GPRGNNV2Augmented --dataset squirrelF --gpu 1 --logging --log-detail --id-log 1027014501 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAugV2-P1027-sqF"
# python tune.py --model GPRGNNV2Augmented --dataset squirrelF --gpu 0 --logging --log-detail --id-log 1027014502 1>logs/${name}-1.log  2>logs/${name}-1.err &
# sleep 3

# name="gprAugV2-randP-sqF"
# python tune.py --model GPRGNNV2Augmented --random-perturb --dataset squirrelF --gpu 1 --logging --log-detail --id-log 1027014503 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAugV2-randP-sqF"
# python tune.py --model GPRGNNV2Augmented --random-perturb --dataset squirrelF --gpu 0 --logging --log-detail --id-log 1027014504 1>logs/${name}-1.log  2>logs/${name}-1.err &
# sleep 3


# name="gprAugV2-P1027-chamF"
# python tune.py --model GPRGNNV2Augmented --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1027014505 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAugV2-randP-chamF"
# python tune.py --model GPRGNNV2Augmented --random-perturb --dataset chameleonF --gpu 1 --logging --log-detail --id-log 1027014506 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="optAug-P1027-chamF"
# python tune.py --model OptBasisAugmented --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1027014507 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="optAug-randP-chamF"
# python tune.py --model OptBasisAugmented --random-perturb --dataset chameleonF --gpu 1 --logging --log-detail --id-log 1027014508 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="optAug-P1027-sqF"
# python tune.py --model OptBasisAugmented --dataset squirrelF --gpu 0 --logging --log-detail --id-log 1027014507 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="optAug-randP-sqF"
# python tune.py --model OptBasisAugmented --random-perturb --dataset squirrelF --gpu 1 --logging --log-detail --id-log 1027014508 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3


# name="optAug-P1028-chamF"
# python tune.py --model OptBasisAugmented --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1028014501 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="optAug1028-randP-chamF"
# python tune.py --model OptBasisAugmented --random-perturb --dataset chameleonF --gpu 1 --logging --log-detail --id-log 1028014502 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="optAug-P1028-sqF"
# python tune.py --model OptBasisAugmented --dataset squirrelF --gpu 0 --logging --log-detail --id-log 1028014503 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="optAug1028-randP-sqF"
# python tune.py --model OptBasisAugmented --random-perturb --dataset squirrelF --gpu 1 --logging --log-detail --id-log 1028014504 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAugV2-P1028-chamF"
# python tune.py --model GPRGNNV2Augmented --dataset chameleonF --gpu 0 --logging --log-detail --id-log 1028014505 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAugV2-randP1028-chamF"
# python tune.py --model GPRGNNV2Augmented --random-perturb --dataset chameleonF --gpu 1 --logging --log-detail --id-log 1028014506 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAugV2-P1028-sqF"
# python tune.py --model GPRGNNV2Augmented --dataset squirrelF --gpu 1 --logging --log-detail --id-log 1028014507 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="gprAugV2-randP1028-sqF"
# python tune.py --model GPRGNNV2Augmented --random-perturb --dataset squirrelF --gpu 1 --logging --log-detail --id-log 1028014508 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

name="Favard-roman-empire"
python tune.py  --dataset roman-empire --model FavardGNN --optuna-n-trials 100 --log-detail --logging --log-detailedCh --gpu 1 --id-log 1114014501 1>logs/${name}.log  2>logs/${name}.err &
sleep 3

name="Favard-minesweeper"
python tune.py  --dataset minesweeper   --model FavardGNN --optuna-n-trials 100 --log-detail --logging --log-detailedCh --gpu 1 --id-log 1114014502 1>logs/${name}.log  2>logs/${name}.err &
sleep 3

name="Favard-tolokers"
python tune.py  --dataset tolokers   --model FavardGNN --optuna-n-trials 100 --log-detail --logging --log-detailedCh --gpu 1 --id-log 1114014503 1>logs/${name}.log  2>logs/${name}.err &
sleep 3

# name="Favard-cs"
# python tune.py  --dataset cs   --model FavardGNN --optuna-n-trials 100 --log-detail --logging --log-detailedCh --gpu 1 --id-log 1114014504 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="Favard-physics"
# python tune.py  --dataset physics   --model FavardGNN --optuna-n-trials 100 --log-detail --logging --log-detailedCh --gpu 1 --id-log 1114014505 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

name="OptBasis-roman-empire"
python tune.py  --dataset roman-empire --model OptBasisGNN --optuna-n-trials 100 --log-detail --logging --log-detailedCh --gpu 1 --id-log 1114014506 1>logs/${name}.log  2>logs/${name}.err &
sleep 3

name="OptBasis-minesweeper"
python tune.py  --dataset minesweeper   --model OptBasisGNN --optuna-n-trials 100 --log-detail --logging --log-detailedCh --gpu 0 --id-log 1114014507 1>logs/${name}.log  2>logs/${name}.err &
sleep 3

name="OptBasis-tolokers"
python tune.py  --dataset tolokers   --model OptBasisGNN --optuna-n-trials 100 --log-detail --logging --log-detailedCh --gpu 0 --id-log 1114014508 1>logs/${name}.log  2>logs/${name}.err &
sleep 3

# name="OptBasis-cs"
# python tune.py  --dataset cs   --model OptBasisGNN --optuna-n-trials 100 --log-detail --logging --log-detailedCh --gpu 0 --id-log 1114014509 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3

# name="OptBasis-physics"
# python tune.py  --dataset physics   --model OptBasisGNN --optuna-n-trials 100 --log-detail --logging --log-detailedCh --gpu 0 --id-log 1114014510 1>logs/${name}.log  2>logs/${name}.err &
# sleep 3