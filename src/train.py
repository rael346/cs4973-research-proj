from lightning import Trainer
from torch.utils.data import DataLoader
from wrapper import CLIPWrapper
from annotation_dataset import AnnotationDataset
from lightning.pytorch.callbacks import LearningRateMonitor
import argparse

PATH_APPAREL_TRAIN_ANNOTATION = './dataset/apparel_train_annotation.csv'
PATH_IMAGES_ANNOTATION = './images/annotation/'
MODEL_BASE = 'ViT-B/32'
MODEL_MEDIUM = 'ViT-B/16'
MODEL_LARGE = 'ViT-L/14'

def train_model(adamw: bool, loss_func: int, lr: float, model: str):
    if model == "base":
        model_name = MODEL_BASE
    elif model == "medium":
        model_name = MODEL_MEDIUM
    elif model == "large":
        model_name = MODEL_LARGE

    model = CLIPWrapper(model_name, adamw, loss_func, lr)
    
    # lr_monitor = LearningRateMonitor(logging_interval="step")
    
    dataset = AnnotationDataset(
        PATH_APPAREL_TRAIN_ANNOTATION, PATH_IMAGES_ANNOTATION, model_name)
    dataloader = DataLoader(dataset, batch_size=100, num_workers=24)
    
    # trainer = Trainer(accelerator='gpu', devices=1, max_epochs=32, callbacks=[lr_monitor])
    trainer = Trainer(accelerator='gpu', devices=1, max_epochs=32)
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model using the annotation set')
    parser.add_argument('--model', type=str, default="base")
    parser.add_argument('--adamw', action="store_true")
    parser.add_argument('--lostfunc', type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    train_model(args.adamw, args.lostfunc, args.lr, args.model)
