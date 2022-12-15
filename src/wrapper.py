from lightning import LightningModule
import torch.nn.functional as F
import clip
import torch
from torch import optim

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class CLIPWrapper(LightningModule):
    def __init__(self, model_name: str, adam_w: bool, loss_func: int, lr: float):
        """A wrapper around the CLIP model using Pytorch Lightning Module for training

        Args:
            model_name (str): The model size to use 
            adam_w (bool): Use the AdamW optimizer
            loss_func (int): The loss function to use 
                0: feedback = target
                1: src + feedback = target
                2: triplet loss with non-target image (WIP)
            lr (float): The learning rate for optimizer
        """
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device, False)
        self.adam_w = adam_w
        self.lr = lr

        self.loss_func = loss_func
        if self.loss_func == 0:
            func_name = "Mpping feedback to target image"
        elif self.loss_func == 1:
            func_name = "Mapping src + feedback to target image"
        print("Using loss function:", func_name, "\n")

    def training_step(self, batch, batch_idx):
        src, feedback, tgt, non_tgt = batch
        
        if self.loss_func == 0:
            logits_per_image, logits_per_text = self.model(tgt, feedback)
            ground_truth = torch.arange(len(feedback), dtype=torch.long, device=feedback.device)
            loss = (F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)) / 2
        
        if self.loss_func == 1:
            # The source embeddings is source image + feedback
            src_embs = F.normalize(self.model.encode_image(src) + self.model.encode_text(feedback), dim=1)
            tgt_embs = F.normalize(self.model.encode_image(tgt), dim=1)
        
            ground_truth = torch.arange(len(feedback), dtype=torch.long, device=feedback.device)
            logits = src_embs @ tgt_embs.t() * self.model.logit_scale.exp()
            loss = (F.cross_entropy(logits, ground_truth) + F.cross_entropy(logits.t(), ground_truth)) / 2
        
        # TODO: experiment with the loss function 
        # if self.loss_func == 2:
        #     src_embs = F.normalize(self.model.encode_image(src) + self.model.encode_text(feedback), dim=1)
        #     tgt_embs = F.normalize(self.model.encode_image(tgt), dim=1)
        #     non_tgt_embs = F.normalize(self.model.encode_image(non_tgt), dim=1)

        #     loss = F.triplet_margin_loss(src_embs, tgt_embs, non_tgt_embs)
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.adam_w:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr,
                        betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                                        betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
        
        return optimizer