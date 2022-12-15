from lightning import LightningModule
import torch.nn.functional as F
import torch.nn as nn
import clip
import torch
from torch import optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class CLIPWrapper(LightningModule):
    def __init__(self, model_name: str, adam_w: bool, loss_func: int, lr: float) -> None:
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device, False)
        self.adam_w = adam_w
        self.loss_func = loss_func
        self.lr = lr
        if self.loss_func == 0:
            func_name = "mapping feedback to target image"
        elif self.loss_func == 1:
            func_name = "mapping src + feedback to target image"
        print("Using loss function:", func_name)

    # Sourced from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    # @property
    # def num_training_steps(self) -> int:
    #     """Total training steps inferred from datamodule and devices."""
    #     dataset = self.trainer
    #     if self.trainer.max_steps:
    #         return self.trainer.max_steps

    #     dataset_size = len(dataset)

    #     num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
    #     if self.trainer.tpu_cores:
    #         num_devices = max(num_devices, self.trainer.tpu_cores)

    #     effective_batch_size = dataset.batch_size * \
    #         self.trainer.accumulate_grad_batches * num_devices
    #     return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    def training_step(self, batch, batch_idx):
        src, feedback, tgt, non_tgt = batch
        
        if self.loss_func == 0:
            logits_per_image, logits_per_text = self.model(tgt, feedback)
            ground_truth = torch.arange(len(feedback), dtype=torch.long, device=feedback.device)
            loss = (F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)) / 2
        
        if self.loss_func == 1:
            src_embs = F.normalize(self.model.encode_image(src) + self.model.encode_text(feedback), dim=1)
            tgt_embs = F.normalize(self.model.encode_image(tgt), dim=1)
        
            ground_truth = torch.arange(len(feedback), dtype=torch.long, device=feedback.device)
            logits = src_embs @ tgt_embs.t() * self.model.logit_scale.exp()
            loss = (F.cross_entropy(logits, ground_truth) + F.cross_entropy(logits.t(), ground_truth)) / 2
            
        # if self.loss_func == 2:
        #     # src_embs = F.normalize(self.model.encode_image(src), dim=1)
        #     feedback_embs = F.normalize(self.model.encode_text(feedback), dim=1)
        #     tgt_embs = F.normalize(self.model.encode_image(tgt), dim=1)
        #     non_tgt_embs = F.normalize(self.model.encode_image(non_tgt), dim=1)
            
        #     loss = F.triplet_margin_loss(feedback_embs, tgt_embs, non_tgt_embs)
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.adam_w:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr,
                        betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                                        betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2000)
        # lr_scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer,
        #     first_cycle_steps=self.num_training_steps,
        #     cycle_mult=1.0,
        #     max_lr=self.lr,
        #     min_lr=0,
        #     warmup_steps=2000
        # )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]