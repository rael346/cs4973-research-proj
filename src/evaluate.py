import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import datetime
from generate_embs import get_feedback_emb_from_query
import clip

PATH_QUERY_FILE = './dataset/query_file_released.jsonl'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def evaluate_query(query_json_path: str, img_emb_json_path: str, feedback_emb_json_path: str, output_path: str):
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
    feedback_embs = pd.read_json(feedback_emb_json_path, lines=True)
    
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
            feedback_emb = torch.tensor(feedback_embs[source_pid])
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
                # torch.tensor
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
    model, preprocess = clip.load('ViT-B/32', device = device)
    epoch1 = "results/models/finetuned/model_epoch1_20221205-123340.pt"
    epoch31 = "results/models/finetuned/model_epoch31_20221205-133726.pt"
    epoch16 = "models/finetuned/model_epoch16_20221206-174028.pt"
    checkpoint = torch.load(epoch16)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    IMG_EMBS_PATH = "results/query/img_embs_b32_clip_finetune_512_epoch_16.jsonl"
    FEEDBACK_EMBS_PATH = "results/query/feedback_embs_b32_clip_finetune_512_epoch_16.jsonl"
    PATH_RESULTS_SAVE = './results/scored_query_file' + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_epoch_1.jsonl'
    
    evaluate_query(PATH_QUERY_FILE, IMG_EMBS_PATH, FEEDBACK_EMBS_PATH, PATH_RESULTS_SAVE)

    
