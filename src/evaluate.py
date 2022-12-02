import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import datetime
from generate_embs import get_feedback_emb_from_query

PATH_QUERY_FILE = './dataset/query_file_released.jsonl'

def evaluate_query(model: SentenceTransformer, query_json_path: str, img_emb_json_path: str, output_path: str):
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
    img_embs = pd.read_json(img_emb_json_path, lines=True)
    print("Getting feedback Embeddings...")
    feedback_embs = get_feedback_emb_from_query(model, query_json_path)

    # Keeping track of missing images (corrupted data)
    missing_img_source = set()
    missing_img_candidate = set()

    # For each query, calculate the cosine similarity between the source emb and the candidates
    for i_row, row in tqdm(query_df.iterrows(), "Query Caculated:"):
        source_pid = row["source_pid"]
        source_img_emb = img_embs.get(source_pid, None)

        # Checking if the source image embeddings is there
        if source_img_emb is None:
            missing_img_source.add(source_pid)
            source_emb = None
        else:
            feedback_emb = feedback_embs[source_pid]
            source_img_emb = torch.tensor(source_img_emb)

            # Add the source image and the corresponding feedback
            source_emb = source_img_emb + feedback_emb

        for i_c, c in enumerate(row["candidates"]):
            c_pid = c["candidate_pid"]
            c_emb = img_embs.get(c_pid, None)

            if c_emb is None:
                missing_img_candidate.add(c_pid)

            # If either the candidate or the source embedding is missing, the score is 0
            if c_emb is None or source_emb is None:
                score = 0
            else:
                score = util.cos_sim(source_emb, c_emb).item()

            query_df_scored.iloc[i_row]['candidates'][i_c]['score'] = score

    print("\nMissing", len(missing_img_source), "source images")
    print(missing_img_source)

    print("\nMissing", len(missing_img_candidate), "candidate images")
    print(missing_img_candidate)

    # return query_df_scored
    query_df_scored.to_json(path_or_buf=output_path,
                            orient='records', lines=True)


if __name__ == "__main__":
    model = SentenceTransformer('clip-ViT-L-14')
    IMG_EMBS_PATH = "results/query/img_embs_l32_no_finetune_512.jsonl"

    PATH_RESULTS_SAVE = './results/scored_query_file' + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.jsonl'
    evaluate_query(model, PATH_QUERY_FILE, IMG_EMBS_PATH, PATH_RESULTS_SAVE)
