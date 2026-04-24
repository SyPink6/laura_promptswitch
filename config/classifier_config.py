import argparse
import os

from modules.basic_utils import deletedir, mkdirp


class ClassifierConfig:
    def __init__(self):
        args = self.parse_args()

        self.dataset_name = args.dataset_name
        self.videos_dir = args.videos_dir
        self.train_manifest = args.train_manifest
        self.val_manifest = args.val_manifest
        self.num_frames = args.num_frames
        self.num_test_frames = args.num_test_frames
        self.num_prompts = args.num_prompts
        self.video_sample_type = args.video_sample_type
        self.video_sample_type_test = args.video_sample_type_test
        self.input_res = args.input_res

        self.use_ema = args.use_ema
        self.model_ema_decay = args.model_ema_decay

        self.exp_name = args.exp_name
        self.model_path = args.model_path
        self.output_dir = args.output_dir
        self.save_every = args.save_every
        self.log_step = args.log_step
        self.evals_per_epoch = args.evals_per_epoch
        self.load_epoch = args.load_epoch

        self.embed_dim = args.embed_dim
        self.temporal_pooling = args.temporal_pooling
        self.num_classes = args.num_classes
        self.classifier_dropout = args.classifier_dropout
        self.freeze_vision_backbone = args.freeze_vision_backbone

        self.clip_lr = args.clip_lr
        self.noclip_lr = args.noclip_lr
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_epochs = args.num_epochs
        self.weight_decay = args.weight_decay
        self.warmup_proportion = args.warmup_proportion

        self.num_workers = args.num_workers
        self.seed = args.seed
        self.no_tensorboard = args.no_tensorboard
        self.tb_log_dir = args.tb_log_dir

    def parse_args(self):
        parser = argparse.ArgumentParser(description="PromptSwitch Visual Binary Classification")

        parser.add_argument("--dataset_name", type=str, default="DeepfakeBinary")
        parser.add_argument("--videos_dir", type=str, default=".", help="Root directory for relative paths in manifests")
        parser.add_argument("--train_manifest", type=str, required=True, help="CSV/JSON/JSONL manifest for training")
        parser.add_argument("--val_manifest", type=str, required=True, help="CSV/JSON/JSONL manifest for validation")
        parser.add_argument("--num_frames", type=int, default=6)
        parser.add_argument("--num_test_frames", type=int, default=12)
        parser.add_argument("--num_prompts", type=int, default=6)
        parser.add_argument("--video_sample_type", default="rand", help="'rand'/'uniform'")
        parser.add_argument("--video_sample_type_test", default="uniform", help="'rand'/'uniform'")
        parser.add_argument("--input_res", type=int, default=224)

        parser.add_argument("--use_ema", action="store_true", default=False)
        parser.add_argument("--model_ema_decay", type=float, default=0.9999)

        parser.add_argument("--exp_name", type=str, required=True)
        parser.add_argument("--output_dir", type=str, default="./outputs_cls")
        parser.add_argument("--save_every", type=int, default=1)
        parser.add_argument("--log_step", type=int, default=10)
        parser.add_argument("--evals_per_epoch", type=int, default=1)
        parser.add_argument("--load_epoch", type=int, help="Epoch to load, or -1 to load model_best.pth")

        parser.add_argument("--embed_dim", type=int, default=512)
        parser.add_argument("--temporal_pooling", type=str, default="avg", choices=["avg", "max"])
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--classifier_dropout", type=float, default=0.0)
        parser.add_argument("--freeze_vision_backbone", action="store_true", default=False,
                            help="Freeze CLIP visual backbone except prompt embedding, prompt attention and classifier")

        parser.add_argument("--clip_lr", type=float, default=1e-6)
        parser.add_argument("--noclip_lr", type=float, default=1e-5)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--test_batch_size", type=int, default=8)
        parser.add_argument("--num_epochs", type=int, default=5)
        parser.add_argument("--weight_decay", type=float, default=0.2)
        parser.add_argument("--warmup_proportion", type=float, default=0.1)

        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--seed", type=int, default=24)
        parser.add_argument("--no_tensorboard", action="store_true", default=False)
        parser.add_argument("--tb_log_dir", type=str, default="logs_cls")

        args = parser.parse_args()

        if args.num_prompts <= 0:
            raise ValueError("--num_prompts must be a positive integer.")
        if args.num_frames % args.num_prompts != 0:
            raise ValueError("--num_frames must be divisible by --num_prompts.")
        if args.num_test_frames % args.num_prompts != 0:
            raise ValueError("--num_test_frames must be divisible by --num_prompts.")
        if args.num_classes != 2:
            raise ValueError("This minimal classifier currently expects --num_classes 2.")

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        mkdirp(args.model_path)
        deletedir(args.tb_log_dir)
        mkdirp(args.tb_log_dir)

        return args
