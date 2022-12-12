from model import Model

if __name__ == "__main__":
    MODEL_BASE = 'ViT-B/32'
    MODEL_MEDIUM = 'ViT-B/16'
    MODEL_LARGE = 'ViT-L/14'

    
    # base result
    # result_folder = "results/base/"
    # checkpoint = ""
    
    # finetune with Adam optimizer (lr = 5e-5)
    # result_folder = "results/finetune_0/"
    # checkpoint = "lightning_logs/version_5/checkpoints/epoch=31-step=3200.ckpt"
    
    # finetune with AdamW optimizer (lr = 1e-7)
    result_folder = "results/finetune_1/"
    checkpoint = "lightning_logs/version_6/checkpoints/epoch=31-step=3200.ckpt"
    
    # finetune with Adam optimizer (change loss function)
    # result_folder = "results/finetune_2/"
    # checkpoint = "lightning_logs/version_7/checkpoints/epoch=31-step=3200.ckpt"
    
    model = Model(MODEL_BASE, checkpoint)
    GALLERY_EMBS_PATH = f"{result_folder}gallery_embs_b32_clip_512.jsonl"
    PATH_QUERY_FILE = './dataset/query_file_released.jsonl'
    QUERY_EMBS_PATH = f"{result_folder}query_embs_b32_clip_512.jsonl"
    RESULT = f"{result_folder}score.jsonl"
    
    model.encode_gallery("images/query/", GALLERY_EMBS_PATH)
    model.encode_query(GALLERY_EMBS_PATH, PATH_QUERY_FILE, QUERY_EMBS_PATH)
    model.calculate_rankings(PATH_QUERY_FILE, QUERY_EMBS_PATH, GALLERY_EMBS_PATH, RESULT)