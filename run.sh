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

name="opt-genius-correct-0"
python tune.py --model NormalNN --dataset genius --gpu 0 --logging --log-detail --id-log 1015014501 1>logs/${name}.log  2>logs/${name}.err &


sleep 3
name="opt-genius-correct-1"
python tune.py --model NormalNN --dataset genius --gpu 1 --logging --log-detail --id-log 1015014501 1>logs/${name}.log  2>logs/${name}.err &