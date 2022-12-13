import clip
import torch
import pandas as pd
from tqdm import tqdm
import json
import os
from PIL import Image
from sentence_transformers import util

class Model:
    def __init__(self, model_name: str, checkpoint_path: str) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Using", model_name, "with checkpoint from", checkpoint_path)
        self.model, self.preprocess = clip.load(model_name, self.device, False)
        
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path)
            state_dict = ckpt["state_dict"]
            
            for key in list(state_dict):
                state_dict[key.replace("model.", "")] = state_dict.pop(key)
                
            self.model.load_state_dict(state_dict)
            
    def encode_query(self, encoded_gallery_path: str, query_path: str, output_path: str):
        query_df = pd.read_json(query_path, lines=True)
        img_embs = pd.read_json(encoded_gallery_path, lines=True)
        emb_dict = {}
        missing_img_source = set()
    
        for _, row in tqdm(query_df.iterrows(), "Query calculated", total=len(query_df)):
            pid = row['source_pid']
            source_img_emb = img_embs.get(pid, None)
            fb = " ".join([row["feedback1"], row["feedback2"], row["feedback3"]])
            text = clip.tokenize(fb).to(self.device)
            
            # text = clip.tokenize([row['feedback1'], row['feedback2'], row['feedback3']]).to(self.device)
            
            if source_img_emb is None:
                missing_img_source.add(pid)
            else:
                with torch.no_grad():
                    text_features = self.model.encode_text(text)
                    src_features = torch.tensor(source_img_emb).to(self.device)
                    
                    query_features = src_features + text_features
                    # query_features = src_features + sum(text_features)
                    emb_dict[pid] = query_features[0].tolist()

        with open(output_path, "w") as outfile:
            json.dump(emb_dict, outfile)
            print("Encoded query saved to", output_path)
            
    def encode_gallery(self, gallery_path: str, output_path: str):
        img_names = os.listdir(gallery_path)
        img_emb_dict = {}
        
        for img_name in tqdm(img_names, desc="Encoding images", total=len(img_names)):
            image = self.preprocess(Image.open(gallery_path + img_name)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                img_emb_dict[img_name.split(".")[0]] = image_features.tolist()[0]
        
        with open(output_path, "w") as outfile:
            json.dump(img_emb_dict, outfile)
            print("Encoded gallery saved to", output_path)
            
    def calculate_rankings(self, query_path: str, query_embs_path: str, encoded_gallery_path: str, output_path: str):
        # read the query file and create a copy of it for appending the score
        query_df = pd.read_json(query_path, lines=True)
        query_df_scored = query_df.copy(deep=True)

        # get the image and feedback embeddings
        img_embs = pd.read_json(encoded_gallery_path, lines=True)
        query_embs = pd.read_json(query_embs_path, lines=True)
        
        # Keeping track of missing images (corrupted data)
        # missing_img_source = set()
        missing_img_candidate = set()

        # For each query, calculate the cosine similarity between the source emb and the candidates
        for i_row, row in tqdm(query_df.iterrows(), "Query score caculated", len(query_df)):
            pid = row["source_pid"]
            query = query_embs.get(pid, None)
            for i_can, can in enumerate(row["candidates"]):
                c_pid = can["candidate_pid"]
                c_emb = img_embs.get(c_pid, None)

                if c_emb is None:
                    missing_img_candidate.add(c_pid)

                # If either the candidate or the source embedding is missing, the score is 0
                if c_emb is None or query is None:
                    score = 0
                else:
                    score = util.cos_sim(query, c_emb).item()
                query_df_scored.iloc[i_row]['candidates'][i_can]['score'] = score

        # print("\nMissing", len(missing_img_source), "source images")
        # print(missing_img_source)

        print("\nMissing", len(missing_img_candidate), "candidate images")
        print(missing_img_candidate)

        # return query_df_scored
        query_df_scored.to_json(path_or_buf=output_path,
                                orient='records', lines=True)
        print("Query rankings saved to", output_path)