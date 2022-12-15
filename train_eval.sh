#! /bin/sh

# Testing learning rate 
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


# Testing new lost function
echo "Train with lost func src + feedback = target, AdamW optimizer, learning rate 5e-5" # This one failed and produced nan loss
python3 src/train.py --adamw --lostfunc 1 --lr 5e-5
python3 src/eval.py --ckpt 4

echo "Train with lost func src + feedback = target, AdamW optimizer, learning rate 1e-7" # Best result so far 
python3 src/train.py --adamw --lostfunc 1 --lr 1e-7
python3 src/eval.py --ckpt 5

# Test to see if the AdamW optimizer and regular Adam optimizer will change the result by much
echo "Train with lost func src + feedback = target, Adam optimizer, learning rate 1e-7"
python3 src/train.py --lostfunc 1 --lr 1e-7
python3 src/eval.py --ckpt 6

# Test on large model 
echo "Evaluate large model performance (no finetune)"
python3 src/train.py --adamw --lostfunc 1 --lr 1e-7 --model large 
python3 src/eval.py --ckpt 7 --model large

echo "Evaluate large model performance (no finetune)"
python3 src/eval.py --nofinetune --ckpt 100 --model large

