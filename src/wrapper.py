from lightning import LightningModule
import torch.nn.functional as F
import torch.nn as nn
import clip
import torch
from torch import optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class CLIPWrapper(LightningModule):
    def __init__(self, model_name: str, adam_w: bool = False, loss_func: int = 0) -> None:
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device, False)
        self.adam_w = adam_w
        self.loss_func = loss_func

    # Sourced from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = len(dataset)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = dataset.batch_size * \
            self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    def training_step(self, batch, batch_idx):
        src, feedback, tgt, non_tgt = batch
        if self.loss_func == 0:
            logits_per_image, logits_per_text = self.model(tgt, feedback)
            ground_truth = torch.arange(len(feedback), dtype=torch.long, device=feedback.device)
            loss = (F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)) / 2
        
        if self.loss_func == 1:
            src_embs = F.normalize(self.model.encode_image(src), dim=1)
            feedback_embs = F.normalize(self.model.encode_text(feedback), dim=1)
            tgt_embs = F.normalize(self.model.encode_image(tgt), dim=1)
            
            ground_truth = torch.arange(len(feedback), dtype=torch.long, device=feedback.device)
            logits = (src_embs + feedback_embs) @ tgt_embs.t()
            loss = (F.cross_entropy(logits, ground_truth) + F.cross_entropy(logits.t(), ground_truth)) / 2
        
        self.log("train_loss", loss)
        if torch.isnan(loss).any():
            print(loss, flush=True)
            
        return loss

    def configure_optimizers(self):
        if not self.adam_w:         
            optimizer = optim.Adam(self.model.parameters(), lr=5e-5,
                                    betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
            return optimizer

        optimizer = optim.AdamW(self.model.parameters(), lr=5e-4,
                        betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

        return optimizer