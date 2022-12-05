import pandas as pd
from sentence_transformers import util
import torch
from tqdm import tqdm
import datetime
import json

PATH_QUERY_FILE = './dataset/query_file_released.jsonl'

def evaluate_query(query_json_path: str, img_emb_json_path: str, source_query_emb_path: str, output_path: str):
    """Evaluate a given query using a given a model and output it to a given file path

    Args:
        model (SentenceTransformer): The given sentence transformer model
        query_json_path (str): The query file path 
        img_emb_json_path (str): The image embeddings file path 
                                 (The same as the output path for get_img_emb)
        output_path (str): The output file path for the new query with score for each candidate
    """
    # read the query file and create a copy of it for appending the score
    query_df = pd.read_json(query_json_path, lines=True)
    query_df_scored = query_df.copy(deep=True)

    # get the image and feedback embeddings
    with open(img_emb_json_path) as f:
        img_embs = json.load(f)

    with open(source_query_emb_path) as f:
        source_embs = json.load(f)

    # For each query, calculate the cosine similarity between the source emb and the candidates
    for i_row, row in tqdm(query_df.iterrows(), "Query Caculated", len(query_df)):
        source_pid = row["source_pid"]
        source_emb = source_embs[source_pid]

        for i_c, c in enumerate(row["candidates"]):
            c_pid = c["candidate_pid"]
            c_emb = img_embs.get(c_pid, None)

            # If either the candidate or the source embedding is missing, the score is 0
            if c_emb is None or source_emb is None:
                score = 0
            else:
                source_emb = torch.Tensor(source_emb)
                c_emb = torch.Tensor(c_emb)
                score = util.cos_sim(source_emb, c_emb).item()

            query_df_scored.iloc[i_row]['candidates'][i_c]['score'] = score

    # return query_df_scored
    query_df_scored.to_json(path_or_buf=output_path,
                            orient='records', lines=True)


if __name__ == "__main__":
    dim = 512
    PATH_RESULTS_SAVE = './results/scored_query_file' + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.jsonl'
    source_query_path = f"results/query/source_query_embs_b32_no_finetune_{dim}.jsonl"
    IMG_EMBS_PATH = f"results/query/img_embs_b32_no_finetune_{dim}.jsonl"
    evaluate_query(PATH_QUERY_FILE, IMG_EMBS_PATH, source_query_path, PATH_RESULTS_SAVE)
