from torch.utils.data import Dataset
import pandas as pd
import os
import clip
from PIL import Image
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class AnnotationDataset(Dataset):
    def __init__(self, annotation_path: str, annotation_images_path: str, model_name: str):
        """Represent the annotation dataset given by the host

        Args:
            annotation_path (str): The annotation csv file given by the host
            annotation_images_path (str): The downloaded images folder from the csv file
            model_name (str): The model name (use for preprocessing)
        """
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

    def clean_up(self):
        """Remove the corrupted annotations (source, target or non-target images not in the downloaded folder)
        """
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

    def __len__(self):
        return len(self.annotations)
    
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
