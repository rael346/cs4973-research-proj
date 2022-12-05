from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import pandas as pd
import math
import os
from PIL import Image
import copy

PATH_IMAGES_ANNOTATION = './images/annotation/'
missing_img_targets = set()
missing_img_non_targets = set()


def finetune(annotation_path, model_name, model_save_path, train_batch_size=16, num_epochs=4):
    model = SentenceTransformer(model_name)
    train_samples = []
    annotation_df = pd.read_csv(annotation_path)

    target_ids = []
    non_target_ids = []
    feedbacks = []

    print("Processing annotation file...")
    for _, row in annotation_df.iterrows():
      target_ids.append(row['Target Image ID'])
      non_target_ids.append(row['Non-Target Image ID'])
      feedbacks.append(
          [row["Feedback 1"], row["Feedback 2"], row["Feedback 3"]])
    print("Number of target IDs: " + str(len(target_ids)))
    print("Number of non target IDs: " + str(len(non_target_ids)))
    print("Batches of feedbacks: " + str(len(feedbacks)))

    print("Processing target training images...")
    for (target_id, feedbacks) in zip(target_ids, feedbacks):
      target_path = PATH_IMAGES_ANNOTATION + target_id + '.jpg'
      if os.path.exists(target_path):
          with Image.open(target_path) as target_img:
            for feedback in feedbacks:
                train_samples.append(InputExample(
                    texts=[target_img, feedback], label=1.0))
      else:
        missing_img_targets.add(target_id)

    print("Processing non-target training images...")
    for (non_target_id, feedbacks) in zip(non_target_ids, feedbacks):
      non_target_path = PATH_IMAGES_ANNOTATION + non_target_id + '.jpg'
      if os.path.exists(non_target_path):
          with Image.open(non_target_path) as non_target_img:
            for feedback in feedbacks:
                train_samples.append(InputExample(
                    texts=[non_target_img, feedback], label=0.0))
      else:
          missing_img_non_targets.add(non_target_id)

    print("Missing", len(missing_img_targets), "target images")
    if len(missing_img_targets) > 0:
        print(missing_img_targets)

    print("Missing", len(missing_img_non_targets), "non target images")
    if len(missing_img_non_targets) > 0:
        print(missing_img_non_targets)

    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    print("Warmup-steps: {}".format(warmup_steps))

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        show_progress_bar=True)
    print("Saved model to " + model_save_path)


if __name__ == "__main__":
    print('CUDA accelerated?: ' + torch.cuda.is_available())
    finetune("./dataset/apparel_train_annotation.csv",
             "clip-ViT-B-32", "results/finetuned")
