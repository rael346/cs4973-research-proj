import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import torch
import datetime
import math
from tqdm import tqdm
import json

PATH_APPAREL_TRAIN_ANNOTATION = './dataset/apparel_train_annotation.csv'
PATH_QUERY_FILE = './dataset/query_file_released.jsonl'
IMAGE_BATCH_SIZE = 3072


def get_img_emb(model: SentenceTransformer, img_path: str) -> dict[str, torch.Tensor]:
    img_names = os.listdir(img_path)
    img_emb_dict = {}

    num_batch = math.ceil(len(img_names) / IMAGE_BATCH_SIZE)
    for i in range(0, len(img_names), IMAGE_BATCH_SIZE):
        print("BATCH", i // IMAGE_BATCH_SIZE + 1, "OF", num_batch)
        batch_img = img_names[i: i + IMAGE_BATCH_SIZE]
        img_embs = model.encode([Image.open(img_path + "/" + name)
                                for name in batch_img], show_progress_bar=True)

        img_emb_dict.update({img_name.split(".")[0]: img_emb.tolist() for img_name,
                             img_emb in zip(batch_img, img_embs)})

    return img_emb_dict


def output_img_emb(model: SentenceTransformer, path: str, output_path: str):
    img_embs = get_img_emb(model, path)
    with open(output_path, "w") as outfile:
        json.dump(img_embs, outfile)


def get_feedback_emb_from_query(model: SentenceTransformer, query_path: str):
    query_df = pd.read_json(query_path, lines=True)
    feedbacks = []
    source_pids = []
    for _, row in query_df.iterrows():
        source_pids.append(row['source_pid'])
        feedbacks.append(row['feedback1'])
        feedbacks.append(row['feedback2'])
        feedbacks.append(row['feedback3'])
    feedback_embs = model.encode(feedbacks, show_progress_bar=True)

    feedback_emb_dict = {}
    for i, pid in enumerate(source_pids):
        feedback1 = feedback_embs[3 * i]
        feedback2 = feedback_embs[3 * i + 1]
        feedback3 = feedback_embs[3 * i + 2]

        feedback_emb = feedback1 + feedback2 + feedback3
        feedback_emb_dict[pid] = feedback_emb

    return feedback_emb_dict


def evaluate_query(model: SentenceTransformer, query_json_path: str, img_emb_json_path: str, output_path: str):
    # read the query file and create a copy of it for appending the score
    query_df = pd.read_json(query_json_path, lines=True)
    query_df_scored = query_df.copy(deep=True)

    # get the image and feedback embeddings
    img_embs = pd.read_json(img_emb_json_path, lines=True)
    print("Getting feedback Embeddings...")
    feedback_embs = get_feedback_emb_from_query(model, PATH_QUERY_FILE)

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
    model = SentenceTransformer('clip-ViT-B-16')
    IMG_EMBS_PATH = "results/query/img_embs_b16_no_finetune_512.jsonl"

    if not os.path.exists(IMG_EMBS_PATH):
        output_img_emb(model, "images/query/", IMG_EMBS_PATH)

    PATH_RESULTS_SAVE = './results/scored_query_file' + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.jsonl'
    evaluate_query(model, PATH_QUERY_FILE, IMG_EMBS_PATH, PATH_RESULTS_SAVE)
