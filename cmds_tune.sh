# name="opt-cham"
# python tune.py --model NormalNN --dataset geom-chameleon --gpu 0 --logging --log-detail --id-log 1011014502 1>>logs/${name}.log  2>>logs/${name}.err &
# sleep 3

# name="gpr-cham"
# python tune.py --model GPRGNN --dataset geom-chameleon --gpu 0 --logging --log-detail --id-log 1011014503 1>>logs/${name}.log  2>>logs/${name}.err &
# sleep 3

name="opt-roman-empire"
python tune.py --model NormalNN --dataset roman-empire --gpu 0 --logging --log-detail --id-log 1015014501 1>logs/${name}.log  2>logs/${name}.err &
sleep 3
