import os
import torch
from torch import nn, optim
import clip
from torch.utils.data import DataLoader
import torch.nn.functional as func
from datasets import Dataset
from PIL import Image
import pandas as pd
import tqdm
import time
import json

PATH_IMAGES_ANNOTATION = './images/annotation/'
PATH_APPAREL_TRAIN_ANNOTATION = './dataset/apparel_train_annotation.csv'
PATH_SAVE_MODEL = './models/finetuned/'

# missing_img_targets = set()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


class Annotation(Dataset):
    # https://github.com/openai/CLIP/issues/83
    def __init__(self, list_src, list_feedbacks, list_target, list_non_target):
        self.src_path = list_src
        self.feedbacks = clip.tokenize(list_feedbacks)
        self.target_path = list_target
        self.non_target_path = list_non_target

    def __len__(self):
        return len(self.feedbacks)

    def __getitem__(self, idx):
        src = preprocess(Image.open(self.src_path[idx]))
        feedback = self.feedbacks[idx]
        target = preprocess(Image.open(self.target_path[idx]))
        non_target = preprocess(Image.open(self.non_target_path[idx]))

        return src, feedback, target, non_target


def convert_models_to_fp32(model):
    # https://github.com/openai/CLIP/issues/57
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def get_image_text_dataset(annotation_path, annotation_images_path):
    # returns a Dataset object containing annotation training target images with feedbacks.
    # Each feedback of the 3 feedbacks are individually paired with their target image
    annotation_df = pd.read_csv(annotation_path)
    # image_paths = []
    # texts = []

    sources = []
    feedbacks = []
    targets = []
    non_targets = []

    num_missing_annotation = 0
    print("Processing annotation file...")
    for _, row in annotation_df.iterrows():
        src_id, target_id, non_target_id = row['Source Image ID'], row['Target Image ID'], row["Non-Target Image ID"]
        fb = ", ".join(
            [row["Feedback 1"], row["Feedback 2"], row["Feedback 3"]])

        src_path = annotation_images_path + src_id + '.jpg'
        target_path = annotation_images_path + target_id + '.jpg'
        non_target_path = annotation_images_path + non_target_id + '.jpg'

        if os.path.exists(src_path) and os.path.exists(target_path) and os.path.exists(non_target_path):
            sources.append(src_path)
            feedbacks.append(fb)
            targets.append(target_path)
            non_targets.append(non_target_path)
        else:
            num_missing_annotation += 1

    # print("Adding target training images to dataset...")
    # for _, (target_id, feedbacks) in enumerate(zip(targets, feedbacks)):
    #     target_path = annotation_images_path + target_id + '.jpg'
    #     if os.path.exists(target_path):
    #         for feedback in feedbacks:
    #             image_paths.append(target_path)
    #             texts.append(feedback)
    #     else:
    #         missing_img_targets.add(target_id)
    # print("Finished adding target training images...")

    print(num_missing_annotation,
          "corrupted annotations (missing src, target or non target images)")

    return Annotation(sources, feedbacks, targets, non_targets)


# Finetunes the model with batch size and number of epochs, saving the
# model state to the specified folder at end of each epoch
def finetune(dataset, path_save_model, train_batch_size=2, num_epochs=1):
    train_dataloader = DataLoader(dataset, batch_size=train_batch_size, num_workers=4)

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    loss_func = nn.MSELoss()

    # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    optimizer = optim.Adam(model.parameters(), lr=5e-5,
                           betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    graph = []
    for epoch in range(num_epochs):
        with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
            for sources, feedbacks, targets, non_targets in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")

                optimizer.zero_grad()

                sources = sources.to(device)
                feedbacks = feedbacks.to(device)
                targets = targets.to(device)
                # non_targets = non_targets.to(device)

                # logits_per_image, logits_per_text = model(images, texts)
                sources_embs = model.encode_image(sources)
                feedbacks_embs = model.encode_text(feedbacks)
                targets_embs = model.encode_image(targets)

                # sources_embs = func.normalize(sources_embs, p = 1, dim=1)
                # feedbacks_embs = func.normalize(feedbacks_embs, p = 1, dim=1)
                # targets_embs = func.normalize(targets_embs, p = 1, dim=1)
                # ground_truth = torch.arange(
                #     len(targets), dtype=torch.long, device=device)

                # total_loss = (loss_img(logits_per_image, ground_truth) +
                #               loss_txt(logits_per_text, ground_truth))/2
                total_loss = loss_func(
                    sources_embs + feedbacks_embs, targets_embs)
                total_loss.backward()

                if device == "cpu":
                    optimizer.step()
                else:
                    # convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)

                tepoch.set_postfix(loss=total_loss.item())

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    },
                   path_save_model + "model_" + f"epoch{str(epoch)}_" + time.strftime("%Y%m%d-%H%M%S") + ".pt")

        graph.append({'epoch': epoch, 'loss': total_loss.tolist()})       
        with open("results/graph/finetune.jsonl", "w") as outfile:
            json.dump(graph, outfile)

if __name__ == "__main__":
    print('CUDA available?: ' + str(torch.cuda.is_available()))
    dataset = get_image_text_dataset(
        PATH_APPAREL_TRAIN_ANNOTATION, PATH_IMAGES_ANNOTATION)
    finetune(dataset, PATH_SAVE_MODEL, train_batch_size=120, num_epochs=32)
