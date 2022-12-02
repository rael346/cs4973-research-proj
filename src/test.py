import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # query_df = pd.read_json("results/query/img_embs_l14_no_finetune_512.jsonl", lines=True)
    # print(len(query_df))
    def dimensionality_reduction(emb_json_path: str, output_path):
        with open(emb_json_path) as f:
            pca = PCA(n_components=128)
            img_embs = json.load(f)
            
            reduced = pca.fit_transform(list(img_embs.values()))
            new_embs = {key : val.tolist() for key, val in zip(img_embs.keys(), reduced)}

            with open(output_path, "w") as outfile:
                json.dump(new_embs, outfile)

    dimensionality_reduction("results/query/img_embs_b32_no_finetune_512.jsonl", "results/query/img_embs_b32_no_finetune_128.jsonl")