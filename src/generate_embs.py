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

    # if os.path.exists(output_path):
    #     f = open(output_path)
    #     img_embs = json.load(f)
    #     calculated_imgs = img_embs.keys()

    #     img_emb_dict.update(img_embs)
    #     for img in calculated_imgs:
    #         img_names.remove(img + ".jpg")
    #     f.close()

    # print("\nImage left to calculate:", len(img_names))
    # print("Image calculated:", len(img_emb_dict), "\n")
    # img_names = list(img_names)

    # Since there is a limit to how many image can be opened
    # at the same time in a system, we will process the images in batch
    # (this batch is different from the batch when doing the model encoding)
    # num_batch = math.ceil(len(img_names) / IMAGE_BATCH_SIZE)

    # for i in range(0, len(img_names), IMAGE_BATCH_SIZE):
    #     print("BATCH", i // IMAGE_BATCH_SIZE + 1, "OF", num_batch)
    #     batch_imgs = img_names[i: i + IMAGE_BATCH_SIZE]
    #     img_embs = model.encode([Image.open(img_path + "/" + name)
    #                             for name in batch_imgs], show_progress_bar=True)

    #     # put the image id and the corresponding embedding into a dictionary
    #     img_emb_dict.update({img_name.split(".")[0]: img_emb.tolist() for img_name,
    #                          img_emb in zip(batch_imgs, img_embs)})

    #     # Write to the given file the dictionary
    #     with open(output_path, "w") as outfile:
    #         json.dump(img_emb_dict, outfile)
    for img_name in tqdm(img_names, desc="Encoding images", total=len(img_names)):
        image = preprocess(Image.open(img_path + img_name)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        img_emb_dict[img_name.split(".")[0]] = image_features.tolist()[0]
    
    with open(output_path, "w") as outfile:
        json.dump(img_emb_dict, outfile)


def get_feedback_emb_from_query(model, query_path: str, output_path: str):
    """Generate the embeddings for the feedbacks from the query using the given model

    Args:
        model (SentenceTransformer): The given sentence transformer model 
        query_path (str): The query file location

    Returns:
        dict: a dictionary that maps the source_pid of the query to the 
        corresponding feedback embeddings
    """
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
        feedback2 = feedback_embs[3 * i]
        feedback3 = feedback_embs[3 * i + 2]

        # Adding all the feedback embeddings together
        feedback_emb = feedback1 + feedback2 + feedback3
        feedback_emb_dict[pid] = feedback_emb

    return feedback_emb_dict

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
    checkpoint = torch.load("results/models/finetuned/model_epoch31_20221205-133726.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if args.query:
        result_folder = "results/query/"
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        img_path = "images/query/"
        img_embs_path = f"{result_folder}img_embs_{version}_finetune_512.jsonl"
        print("Generating query images embeddings...")
        # image = preprocess(Image.open(img_path + "21-cdT54ZSL.jpg")).unsqueeze(0).to(device)
        # image_features = model.encode_image(image)
        # print(len(image_features.tolist()[0]))

    get_img_emb(model, preprocess, img_path, img_embs_path)
    print("DONE!, embeddings are in", img_embs_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate image embeddings')
    parser.add_argument('--annotation', action="store_true")
    parser.add_argument('--query', action="store_true")
    parser.add_argument('--gallery', action="store_true")
    parser.add_argument('--model', type=str, default="base")
    parser.add_argument('--models', action="store_true")
    args = parser.parse_args()
    generate_embs(args)
