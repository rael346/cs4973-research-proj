# CS4973 Research Project: One-shot Product Search

[Link to main challenge](https://eval.ai/web/challenges/challenge-page/1845/overview)

## Setup

### Generate virtual environment

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Download dataset image file

```
python3 src/downloader.py --annotation --query
```

### Train model

Our best model so far (mAP: 0.50)

```
python3 src/train.py --lostfunc 1 --lr 1e-7
```

### Evaluate model (generate a new query with rankings to upload on leaderboard)

1. Baseline (no finetuning, ViT-B/32)

**Note:** since we're using `--nofinetune`, the checkpoint number here doesn't matter as long as we don't repeat it

```
python3 src/eval.py --nofinetune --ckpt 100
```

2. Large model (no finetuning, ViT-L/14)

```
python3 src/eval.py --nofinetune --ckpt 101
```

3. Score for best model in section 3

**Note:** the checkpoint number here is the checkpoint for the `lightning_logs/version_{num}`. So check the version number in `lightning_logs` before evaluating. `eval.py` will create a corresponding `version_{num}/` in `results/`

```
python3 src/eval.py --ckpt 0
```

\*For more iteration in the research, see `train_eval.sh`
