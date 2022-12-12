from lightning import Trainer
from torch.utils.data import DataLoader
from wrapper import CLIPWrapper
from annotation_dataset import AnnotationDataset

if __name__ == "__main__":
    PATH_APPAREL_TRAIN_ANNOTATION = './dataset/apparel_train_annotation.csv'
    PATH_IMAGES_ANNOTATION = './images/annotation/'
    MODEL_BASE = 'ViT-B/32'
    MODEL_MEDIUM = 'ViT-B/16'
    MODEL_LARGE = 'ViT-L/14'

    # regular Adam optimizer, mapping feedback to target image
    # adam_w, loss_func = False, 0
    
    # AdamW optimizer, mapping feedback = target image
    # adam_w, loss_func = True, 0
    
    # regular Adam optimizer, mapping src + feedback = target
    # adam_w, loss_func = False, 1
    
    # AdamW optimizer, mapping src + feedback = target
    adam_w, loss_func = True, 1
    
    model = CLIPWrapper(MODEL_BASE)
    dataset = AnnotationDataset(
        PATH_APPAREL_TRAIN_ANNOTATION, PATH_IMAGES_ANNOTATION, MODEL_BASE)

    dataloader = DataLoader(dataset, batch_size=120, num_workers=24)
    trainer = Trainer(accelerator='gpu', devices=1,
                      limit_train_batches=100, max_epochs=32)
    trainer.fit(model, dataloader)
