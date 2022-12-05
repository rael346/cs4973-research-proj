import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import math
import json
import argparse
import numpy as np
import torch
from sklearn.decomposition import PCA

PATH_APPAREL_TRAIN_ANNOTATION = './dataset/apparel_train_annotation.csv'
PATH_QUERY_FILE = './dataset/query_file_released.jsonl'
IMAGE_BATCH_SIZE = 1024

MODEL_BASE = 'clip-ViT-B-32'
MODEL_MEDIUM = 'clip-ViT-B-16'
MODEL_LARGE = 'clip-ViT-L-14'


def get_img_emb(model: SentenceTransformer, img_path: str, output_path: str):
    """Generate image embeddings for every image in a given folder using the given model
    and output it to the given output file name 

    Args:
        model (SentenceTransformer): The given sentence transformer model
        img_path (str): The image folder path 
        output_path (str): The file to output the image embeddings
    """
    # Get all the image names in the folder
    img_names = set(os.listdir(img_path))
    img_emb_dict = {}

    if os.path.exists(output_path):
        f = open(output_path)
        img_embs = json.load(f)
        calculated_imgs = img_embs.keys()

        img_emb_dict.update(img_embs)
        for img in calculated_imgs:
            img_names.remove(img + ".jpg")
        f.close()

    print("\nImage left to calculate:", len(img_names))
    print("Image calculated:", len(img_emb_dict), "\n")
    img_names = list(img_names)

    # Since there is a limit to how many image can be opened
    # at the same time in a system, we will process the images in batch
    # (this batch is different from the batch when doing the model encoding)
    num_batch = math.ceil(len(img_names) / IMAGE_BATCH_SIZE)

    for i in range(0, len(img_names), IMAGE_BATCH_SIZE):
        print("BATCH", i // IMAGE_BATCH_SIZE + 1, "OF", num_batch)
        batch_imgs = img_names[i: i + IMAGE_BATCH_SIZE]
        img_embs = model.encode([Image.open(img_path + "/" + name)
                                for name in batch_imgs], show_progress_bar=True)

        # put the image id and the corresponding embedding into a dictionary
        img_emb_dict.update({img_name.split(".")[0]: img_emb.tolist() for img_name,
                             img_emb in zip(batch_imgs, img_embs)})

        # Write to the given file the dictionary
        with open(output_path, "w") as outfile:
            json.dump(img_emb_dict, outfile)


def get_source_emb_from_query(model: SentenceTransformer, query_path: str, img_emb_json_path: str, output_path: str):
    """Generate the embeddings for the query using the given model

    Args:
        model (SentenceTransformer): The given sentence transformer model 
        query_path (str): The query file location
        img_emb_json_path (str): The image embeddings for the query 
        output_path (str): The output file path for the generating embeddings

    Returns:
        dict: a dictionary that maps the source_pid of the query to the 
        corresponding feedback embeddings
    """
    with open(img_emb_json_path) as f:
        img_embs = json.load(f)
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

        source_img_emb = img_embs.get(pid, None)
        if source_img_emb is None:
            source_emb = None
        else:
            source_img_emb = torch.tensor(source_img_emb)
            feedback_emb = feedback1 + feedback2 + feedback3
            source_emb = source_img_emb + feedback_emb
            source_emb = source_emb.tolist()

        feedback_emb_dict[pid] = source_emb

    with open(output_path, "w") as outfile:
        json.dump(feedback_emb_dict, outfile)

def generate_embs(agrs):
    if args.model == "base":
        model_name = MODEL_BASE
        version = "b32"
    elif args.model == "med":
        model_name = MODEL_MEDIUM
        version = "b16"
    elif args.model == "large":
        model_name = MODEL_LARGE
        version = "l14"
    else:
        print("No model exists with name:", args.model)
    model = SentenceTransformer(model_name)

    if args.query:
        result_folder = "results/query/"
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        img_path = "images/query/"
        img_embs_path = f"{result_folder}img_embs_{version}_no_finetune_512.jsonl"
        print("Generating query images embeddings...")
        get_img_emb(model, img_path, img_embs_path)
        print("DONE!, embeddings are in", img_embs_path)

    if args.source_query:
        output = "results/query/source_query_embs_b32_no_finetune_512.jsonl"
        get_source_emb_from_query(model, PATH_QUERY_FILE, "results/query/img_embs_b32_no_finetune_512.jsonl", output)
    

def dimensionality_reduction(emb_json_path: str):
    pca = PCA(n_components=128)
    img_embs = pd.read_json(emb_json_path, lines=True)
    pca.fit(img_embs.values)
    pca_comp = np.asarray(pca.components_)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings')
    parser.add_argument('--annotation', action="store_true")
    parser.add_argument('--query', action="store_true")
    parser.add_argument('--gallery', action="store_true")
    parser.add_argument('--model', type=str, default="base")
    parser.add_argument('--source_query', action="store_true")
    args = parser.parse_args()
    generate_embs(args)
