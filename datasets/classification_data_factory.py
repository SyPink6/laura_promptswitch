from torch.utils.data import DataLoader

from datasets.deepfake_dataset import DeepfakeVideoDataset
from datasets.model_transforms import init_transform_dict


class ClassificationDataFactory:
    @staticmethod
    def get_data_loader(config, split_type="train"):
        img_transforms = init_transform_dict(config.input_res)
        train_img_tfms = img_transforms["clip_train"]
        test_img_tfms = img_transforms["clip_test"]

        if split_type == "train":
            dataset = DeepfakeVideoDataset(
                config,
                manifest_path=config.train_manifest,
                split_type="train",
                img_transforms=train_img_tfms,
            )
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
            )

        dataset = DeepfakeVideoDataset(
            config,
            manifest_path=config.val_manifest,
            split_type="val",
            img_transforms=test_img_tfms,
        )
        return DataLoader(
            dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
