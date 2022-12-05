import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # query_df = pd.read_json("results/query/img_embs_l14_no_finetune_512.jsonl", lines=True)
    # print(len(query_df))
    def dimensionality_reduction(img_emb_json_path: str, img_emb_output_path: str, src_query_emb_json_path: str, src_query_emb_output_path: str):
        with open(img_emb_json_path) as f:
            img_embs = json.load(f)

        with open(src_query_emb_json_path) as f:
            src_query_embs = json.load(f)

        pca = PCA(n_components=128)
        filter_src_dict = dict(filter(lambda e: e[1] is not None, src_query_embs.items()))

        fitted_pca = pca.fit(list(img_embs.values()))
        
        src_reduced = fitted_pca.transform(list(filter_src_dict.values()))
        new_src_embs = {key : val.tolist() for key, val in zip(filter_src_dict.keys(), src_reduced)}

        src_query_embs.update(new_src_embs)
        with open(img_emb_output_path, "w") as outfile:
            json.dump(src_query_embs, outfile)

        img_reduced = fitted_pca.transform(list(img_embs.values()))
        new_img_embs = {key : val.tolist() for key, val in zip(img_embs.keys(), img_reduced)}

        with open(src_query_emb_output_path, "w") as outfile:
            json.dump(new_img_embs, outfile)

    img_512 = "results/query/img_embs_b32_no_finetune_512.jsonl"
    src_512 = "results/query/source_query_embs_b32_no_finetune_512.jsonl"

    img_128 = "results/query/img_embs_b32_no_finetune_128.jsonl"
    src_128 = "results/query/source_query_embs_b32_no_finetune_128.jsonl"

    dimensionality_reduction(img_512, img_128, src_512, src_128)