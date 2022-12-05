import os
import torch
from torch import nn, optim
import clip
from torch.utils.data import DataLoader
from datasets import Dataset
from PIL import Image
import pandas as pd
import tqdm
import time

PATH_IMAGES_ANNOTATION = './images/annotation/'
PATH_APPAREL_TRAIN_ANNOTATION = './dataset/apparel_train_annotation.csv'
PATH_SAVE_MODEL = './results/models/finetuned/'

missing_img_targets = set()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# https://github.com/openai/CLIP/issues/83
class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.title = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

# returns a Dataset object containing annotation training target images with feedbacks.
# Each feedback of the 3 feedbacks are individually paired with their target image
def get_image_text_dataset(annotation_path, annotation_images_path):
    annotation_df = pd.read_csv(annotation_path)
    image_paths = []
    texts = []

    target_ids = []
    feedbacks = []

    print("Processing annotation file...")
    for _, row in annotation_df.iterrows():
        target_ids.append(row['Target Image ID'])
        feedbacks.append(
            [row["Feedback 1"], row["Feedback 2"], row["Feedback 3"]])
    print("Number of unique target IDs: " + str(len(set(target_ids))))
    print("Batches of feedbacks: " + str(len(feedbacks)))

    print("Adding target training images to dataset...")
    for _, (target_id, feedbacks) in enumerate(zip(target_ids, feedbacks)):
        target_path = annotation_images_path + target_id + '.jpg'
        if os.path.exists(target_path):
            for feedback in feedbacks:
                image_paths.append(target_path)
                texts.append(feedback)
        else:
            missing_img_targets.add(target_id)
    print("Finished adding target training images...")

    print("Missing", len(missing_img_targets), "target images")
    if len(missing_img_targets) > 0:
        print(missing_img_targets)

    return image_title_dataset(image_paths, texts)


# Finetunes the model with batch size and number of epochs, saving the
# model state to the specified folder at end of each epoch
def finetune(dataset, path_save_model, train_batch_size=2, num_epochs=1):
    train_dataloader = DataLoader(dataset, batch_size=train_batch_size, num_workers=4)

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    optimizer = optim.Adam(model.parameters(), lr=5e-5,
                           betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
                           
    for epoch in range(num_epochs):
        with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
            for images, texts in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")

                optimizer.zero_grad()

                images = images.to(device)
                texts = texts.to(device)

                logits_per_image, logits_per_text = model(images, texts)

                ground_truth = torch.arange(
                    len(images), dtype=torch.long, device=device)

                total_loss = (loss_img(logits_per_image, ground_truth) +
                              loss_txt(logits_per_text, ground_truth))/2
                total_loss.backward()
                if device == "cpu":
                    optimizer.step()
                else:
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)

                tepoch.set_postfix(loss=total_loss.item())

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    },
                   path_save_model + "model_" + f"epoch{str(epoch)}_" + time.strftime("%Y%m%d-%H%M%S") + ".pt")


if __name__ == "__main__":
    print('CUDA available?: ' + str(torch.cuda.is_available()))
    dataset = get_image_text_dataset(
        PATH_APPAREL_TRAIN_ANNOTATION, PATH_IMAGES_ANNOTATION)
    finetune(dataset, PATH_SAVE_MODEL, train_batch_size=100, num_epochs=32)
