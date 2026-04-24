import numpy as np
import torch
import torch.nn as nn

from trainer.base_trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer,
        config,
        train_data_loader,
        valid_data_loader,
        lr_scheduler=None,
        writer=None,
        use_ema=False,
    ):
        super().__init__(
            model=model,
            loss={"cls": nn.CrossEntropyLoss()},
            optimizer=optimizer,
            config=config,
            writer=writer,
            use_ema=use_ema,
        )
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.best_acc = -1.0

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, data in enumerate(self.train_data_loader):
            labels = data["label"].to(self.device)
            data["video"] = data["video"].to(self.device)

            model_output = self.model(data)
            logits = model_output["logits"]
            loss = self.loss["cls"](logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if self.use_ema:
                self.model_ema.update(self.model)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.detach().item()
            self.global_step += 1

            if self.writer is not None:
                self.writer.add_scalar("train/loss_cls", loss.detach().item(), self.global_step)

            if batch_idx % self.log_step == 0:
                curr_acc = (preds == labels).float().mean().item()
                print(
                    "Train Epoch: {} dl: {}/{} Loss Cls: {:.6f} Acc: {:.4f}".format(
                        epoch,
                        batch_idx,
                        len(self.train_data_loader) - 1,
                        loss.detach().item(),
                        curr_acc,
                    )
                )

        train_loss = total_loss / max(len(self.train_data_loader), 1)
        train_acc = total_correct / max(total_samples, 1)

        if self.use_ema:
            model = self.model_ema.module
        else:
            model = self.model
        val_res = self._valid_epoch_step(model, epoch, len(self.train_data_loader) - 1, len(self.train_data_loader) - 1)
        self.model.train()

        if val_res["acc_val"] > self.best_acc:
            self.best_acc = val_res["acc_val"]
            self._save_checkpoint(epoch, save_best=True)

        print(" Current Best Val Acc is {}\n".format(self.best_acc))

        return {
            "loss_train": train_loss,
            "acc_train": train_acc,
            **val_res,
        }

    def _valid_epoch_step(self, model, epoch, step, num_steps):
        model.eval()
        total_val_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data in self.valid_data_loader:
                labels = data["label"].to(self.device)
                data["video"] = data["video"].to(self.device)

                model_output = model(data)
                logits = model_output["logits"]
                loss = self.loss["cls"](logits, labels)

                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_val_loss += loss.detach().item()

        avg_loss = total_val_loss / max(len(self.valid_data_loader), 1)
        acc = total_correct / max(total_samples, 1)

        print(
            "-----Val Epoch: {}, dl: {}/{}-----\n"
            "Loss: {:.6f}\n"
            "Acc: {:.4f}\n"
            "--------------------------------".format(
                epoch,
                step,
                num_steps,
                avg_loss,
                acc,
            )
        )

        if self.writer is not None:
            self.writer.add_scalar("val/loss_cls", avg_loss, self.global_step)
            self.writer.add_scalar("val/acc", acc, self.global_step)

        return {
            "loss_val": avg_loss,
            "acc_val": acc,
        }
