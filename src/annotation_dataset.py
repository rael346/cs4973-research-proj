from torch.utils.data import Dataset
from torchvision import transforms as T
import pandas as pd
import os
import clip
from PIL import Image
import torch
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class AnnotationDataset(Dataset):
    def __init__(self, annotation_path, annotation_images_path, model_name):
        try:
            self.annotations = pd.read_csv(annotation_path)
        except:
            raise Exception(
                "Cannot find annotation file at path:", annotation_path)

        if not os.path.exists(annotation_images_path):
            raise Exception("Cannot find images at path:",
                            annotation_images_path)

        self.images_path = annotation_images_path
        _, self.preprocess = clip.load(model_name, device, False)

        self.clean_up()

    def __len__(self):
        return len(self.annotations)

    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def clean_up(self):
        missing_row = []

        for index, row in self.annotations.iterrows():
            src_id, tgt_id, non_tgt_id = row['Source Image ID'], row['Target Image ID'], row["Non-Target Image ID"]
            src_path = self.images_path + src_id + '.jpg'
            target_path = self.images_path + tgt_id + '.jpg'
            non_target_path = self.images_path + non_tgt_id + '.jpg'

            if not os.path.exists(src_path) or not os.path.exists(target_path) or not os.path.exists(non_target_path):
                missing_row.append(index)

        print("Number of corrupted annotation (source, target or non target images):", len(
            missing_row))
        self.annotations.drop(labels=missing_row, axis=0, inplace=True)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        src_id, tgt_id, non_tgt_id = row['Source Image ID'], row['Target Image ID'], row["Non-Target Image ID"]
        fb = " ".join(
            [row["Feedback 1"], row["Feedback 2"], row["Feedback 3"]])

        src_path = self.images_path + src_id + '.jpg'
        target_path = self.images_path + tgt_id + '.jpg'
        non_target_path = self.images_path + non_tgt_id + '.jpg'

        tokenized_fb = clip.tokenize(fb)[0]

        src_tensor = self.preprocess(Image.open(src_path))
        tgt_tensor = self.preprocess(Image.open(target_path))
        non_tgt_tensor = self.preprocess(Image.open(non_target_path))
        
        return src_tensor, tokenized_fb, tgt_tensor, non_tgt_tensor
