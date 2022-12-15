from model import Model
import os
import argparse

MODEL_BASE = 'ViT-B/32'
MODEL_MEDIUM = 'ViT-B/16'
MODEL_LARGE = 'ViT-L/14'


def evaluate_query(ckpt_ver: int, no_finetune: bool, model: str):
    result_folder = f"results/version_{ckpt_ver}/"

    if not os.path.exists("results/"):
        os.mkdir("results/")

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    checkpoint = "" if no_finetune else f"lightning_logs/version_{ckpt_ver}/checkpoints/epoch=31-step=4736.ckpt"

    if model == "base":
        model_name = MODEL_BASE
        version = "b32"
    elif model == "medium":
        model_name = MODEL_MEDIUM
        version = "b16"
    elif model == "large":
        model_name = MODEL_LARGE
        version = "l14"

    model = Model(model_name, checkpoint)
    GALLERY_EMBS_PATH = f"{result_folder}gallery_embs_{version}.jsonl"
    PATH_QUERY_FILE = './dataset/query_file_released.jsonl'
    QUERY_EMBS_PATH = f"{result_folder}query_embs_{version}.jsonl"
    RESULT = f"{result_folder}query_score.jsonl"

    model.encode_gallery("images/query/", GALLERY_EMBS_PATH)
    model.encode_query(GALLERY_EMBS_PATH, PATH_QUERY_FILE, QUERY_EMBS_PATH)
    model.calculate_rankings(
        PATH_QUERY_FILE, QUERY_EMBS_PATH, GALLERY_EMBS_PATH, RESULT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Query Set')
    parser.add_argument('--ckpt', type=int, required=True)
    parser.add_argument('--nofinetune', action="store_true")
    parser.add_argument('--model', type=str, default="base")
    args = parser.parse_args()
    evaluate_query(args.ckpt, args.nofinetune, args.model)
