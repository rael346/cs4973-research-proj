from model import Model
import os

if __name__ == "__main__":
    MODEL_BASE = 'ViT-B/32'
    MODEL_MEDIUM = 'ViT-B/16'
    MODEL_LARGE = 'ViT-L/14'

    version = 7
    result_folder = f"results/version_{version}/"
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    checkpoint = f"lightning_logs/version_{version}/checkpoints/epoch=31-step=3200.ckpt"
    
    model = Model(MODEL_BASE, checkpoint)
    GALLERY_EMBS_PATH = f"{result_folder}gallery_embs_b32_clip_512.jsonl"
    PATH_QUERY_FILE = './dataset/query_file_released.jsonl'
    QUERY_EMBS_PATH = f"{result_folder}query_embs_b32_clip_512.jsonl"
    RESULT = f"{result_folder}query_score.jsonl"
    
    model.encode_gallery("images/query/", GALLERY_EMBS_PATH)
    model.encode_query(GALLERY_EMBS_PATH, PATH_QUERY_FILE, QUERY_EMBS_PATH)
    model.calculate_rankings(PATH_QUERY_FILE, QUERY_EMBS_PATH, GALLERY_EMBS_PATH, RESULT)