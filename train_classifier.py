import os
import random

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from config.classifier_config import ClassifierConfig
from datasets.classification_data_factory import ClassificationDataFactory
from model.promptswitch_classifier import PromptSwitchClassifier
from trainer.classifier_trainer import ClassificationTrainer


def main():
    config = ClassifierConfig()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_data_loader = ClassificationDataFactory.get_data_loader(config, split_type="train")
    valid_data_loader = ClassificationDataFactory.get_data_loader(config, split_type="val")
    model = PromptSwitchClassifier(config)

    optimizer_grouped_params = []
    if model.clip_params:
        optimizer_grouped_params.append({"params": model.clip_params, "lr": config.clip_lr})
    if model.noclip_params:
        optimizer_grouped_params.append({"params": model.noclip_params, "lr": config.noclip_lr})
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)

    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    trainer = ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=scheduler,
        writer=writer,
        use_ema=config.use_ema,
    )

    if config.load_epoch is not None:
        if config.load_epoch == -1:
            trainer.load_checkpoint("model_best.pth")
        else:
            trainer.load_checkpoint(f"checkpoint-epoch{config.load_epoch}.pth")

    trainer.train()


if __name__ == "__main__":
    main()
