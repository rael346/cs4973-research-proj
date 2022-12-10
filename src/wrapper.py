from lightning import LightningModule
import torch.nn.functional as F
import clip
import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class CLIPWrapper(LightningModule):
    def __init__(self, model_name: str) -> None:
        self.model, self.preprocess = clip.load(model_name, device, False)

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

        src_embs = [F.normalize(self.model.encode_image(s), dim=1)
                    for s in src]
        feedback_embs = [F.normalize(
            self.model.encode_text(f), dim=1) for f in feedback]
        tgt_embs = [F.normalize(self.model.encode_text(t), dim=1) for t in tgt]
        # non_embs = [F.normalize(self.model.encode_text(n), dim=1) for n in non_tgt]

        logits = (torch.cat(src_embs) + torch.cat(feedback_embs)
                  ) @ torch.cat(tgt_embs).t() * self.model.logit_scale.exp()
        ground_truth = torch.arange(len(src_embs)).long().to(src_embs.device)
        loss = (F.cross_entropy(logits, ground_truth) +
                F.cross_entropy(logits.t(), ground_truth)) / 2
        return loss