import pandas as pd
# from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import math
import json
import argparse
import numpy as np
import clip
import torch
from tqdm import tqdm
# from sklearn.decomposition import PCA

PATH_APPAREL_TRAIN_ANNOTATION = './dataset/apparel_train_annotation.csv'
PATH_QUERY_FILE = './dataset/query_file_released.jsonl'
IMAGE_BATCH_SIZE = 1024

MODEL_BASE = 'ViT-B/32'
MODEL_MEDIUM = 'ViT-B/16'
MODEL_LARGE = 'ViT-L/14'

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_img_emb(model, preprocess, img_path: str, output_path: str):
    """Generate image embeddings for every image in a given folder using the given model
    and output it to the given output file name 

    Args:
        model (SentenceTransformer): The given sentence transformer model
        img_path (str): The image folder path 
        output_path (str): The file to output the image embeddings
    """
    # Get all the image names in the folder
    img_names = os.listdir(img_path)
    img_emb_dict = {}
    
    for img_name in tqdm(img_names, desc="Encoding images", total=len(img_names)):
        image = preprocess(Image.open(img_path + img_name)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            img_emb_dict[img_name.split(".")[0]] = image_features.tolist()[0]
    
    with open(output_path, "w") as outfile:
        json.dump(img_emb_dict, outfile)


def get_feedback_emb_from_query(model, query_path: str, feedback_output_path):
    """Generate the embeddings for the feedbacks from the query using the given model

    Args:
        model (SentenceTransformer): The given sentence transformer model 
        query_path (str): The query file location

    Returns:
        dict: a dictionary that maps the source_pid of the query to the 
        corresponding feedback embeddings
    """
    query_df = pd.read_json(query_path, lines=True)
    feedback_emb_dict = {}
    for _, row in tqdm(query_df.iterrows(), "Processing Feedback", total=len(query_df)):
        pid = row['source_pid']
        text = clip.tokenize([row['feedback1'], row['feedback2'], row['feedback3']]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            feedback_emb_dict[pid] = sum(text_features).tolist()

    with open(feedback_output_path, "w") as outfile:
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
    
    if args.models:
        print(clip.available_models())


    model, preprocess = clip.load(model_name, device = device, jit=False) #Must set jit=False for training
    # epoch1 = "results/models/finetuned/model_epoch1_20221205-123340.pt"
    # epoch31 = "results/models/finetuned/model_epoch31_20221205-133726.pt"
    epoch16 = "models/finetuned/model_epoch16_20221206-174028.pt"
    checkpoint = torch.load(epoch16)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if args.query:
        result_folder = "results/query/"
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        img_path = "images/query/"
        img_embs_path = f"{result_folder}img_embs_{version}_clip_finetune_512_epoch_16.jsonl"
        print("Generating query images embeddings...")
        get_img_emb(model, preprocess, img_path, img_embs_path)
        print("DONE!, embeddings are in", img_embs_path)

    if args.feedback:
        result_folder = "results/query/"
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        feedback_embs_path = f"{result_folder}feedback_embs_{version}_clip_finetune_512_epoch_16.jsonl"
        print("Generating query feedback embeddings...")
        get_feedback_emb_from_query(model, PATH_QUERY_FILE, feedback_embs_path)
        print("DONE!, embeddings are in", feedback_embs_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate image embeddings')
    parser.add_argument('--annotation', action="store_true")
    parser.add_argument('--query', action="store_true")
    parser.add_argument('--gallery', action="store_true")
    parser.add_argument('--model', type=str, default="base")
    parser.add_argument('--models', action="store_true")
    parser.add_argument('--feedback', action="store_true")
    args = parser.parse_args()
    generate_embs(args)
