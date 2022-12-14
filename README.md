# CS4973 Research Project: One-shot Product Search

[Link to main challenge](https://eval.ai/web/challenges/challenge-page/1845/overview)

## Setup 

1. Generate virtual environment 

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

2. Download dataset image file

```
python3 src/downloader.py --annotation --query
``` 

3. Generate image embeddings

Note: There are three types of model: `base`, `med`, `large`, corresponding to `clip-ViT-B-32`, `clip-ViT-B-16`, `clip-ViT-L-14`. 

```
python3 src/generate_embs.py --query
```

For bigger model, 
```
python3 src/generate_embs.py --query --model large
```

version 2
loss 1, lr = 1e-7

version 3 
incoporate logit scale

version 4
loss 0, adam optimizer lr = 1e-7