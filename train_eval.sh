#! /bin/sh


echo "Train with lost func feedback = target, Adam optimizer, learning rate 5e-5"
python3 src/train.py --lostfunc 0 --lr 5e-5
python3 src/eval.py --ckpt 0

echo "Train with lost func feedback = target, Adam optimizer, learning rate 1e-7"
python3 src/train.py --lostfunc 0 --lr 1e-7
python3 src/eval.py --ckpt 1

echo "Train with lost func feedback = target, AdamW optimizer, learning rate 5e-5"
python3 src/train.py --adamw --lostfunc 0 --lr 5e-5
python3 src/eval.py --ckpt 2

echo "Train with lost func feedback = target, AdamW optimizer, learning rate 1e-7"
python3 src/train.py --adamw --lostfunc 0 --lr 1e-7
python3 src/eval.py --ckpt 3

echo "Train with lost func src + feedback = target, AdamW optimizer, learning rate 5e-5"
python3 src/train.py --adamw --lostfunc 1 --lr 5e-5
python3 src/eval.py --ckpt 4

echo "Train with lost func src + feedback = target, AdamW optimizer, learning rate 1e-7"
python3 src/train.py --adamw --lostfunc 1 --lr 1e-7
python3 src/eval.py --ckpt 5