import json
import os

import pandas as pd
from torch.utils.data import Dataset

from datasets.video_capture import VideoCapture


def _parse_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()
    label_map = {
        "0": 0,
        "1": 1,
        "real": 0,
        "fake": 1,
        "genuine": 0,
        "spoof": 1,
        "bonafide": 0,
        "bona_fide": 0,
        "deepfake": 1,
        "false": 0,
        "true": 1,
    }
    if text not in label_map:
        raise ValueError(f"Unsupported label value: {value}")
    return label_map[text]


class DeepfakeVideoDataset(Dataset):
    def __init__(self, config, manifest_path, split_type="train", img_transforms=None):
        self.config = config
        self.manifest_path = manifest_path
        self.split_type = split_type
        self.img_transforms = img_transforms
        self.videos_dir = config.videos_dir

        if split_type == "train":
            self.num_frames = config.num_frames
            self.video_sample_type = config.video_sample_type
        else:
            self.num_frames = config.num_test_frames
            self.video_sample_type = config.video_sample_type_test

        self.records = self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path):
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: '{manifest_path}'")

        ext = os.path.splitext(manifest_path)[1].lower()
        if ext == ".csv":
            rows = pd.read_csv(manifest_path).to_dict("records")
        elif ext == ".json":
            with open(manifest_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                rows = payload.get("data", payload.get("items"))
            else:
                rows = payload
        elif ext == ".jsonl":
            rows = []
            with open(manifest_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        else:
            raise ValueError("Manifest must be CSV, JSON, or JSONL.")

        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Manifest '{manifest_path}' is empty or malformed.")

        normalized = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {idx} is not a dictionary.")

            video_key = next(
                (key for key in ["video", "video_path", "path", "filepath", "file"] if key in row),
                None,
            )
            label_key = next(
                (key for key in ["label", "target", "class", "y"] if key in row),
                None,
            )

            if video_key is None or label_key is None:
                raise ValueError(
                    f"Manifest row {idx} must contain a video path field and a label field. "
                    "Accepted video keys: video/video_path/path/filepath/file; "
                    "label keys: label/target/class/y."
                )

            video_path = row[video_key]
            if not os.path.isabs(video_path):
                video_path = os.path.join(self.videos_dir, video_path)

            record_id = row.get("video_id", os.path.splitext(os.path.basename(video_path))[0])
            normalized.append({
                "video_id": str(record_id),
                "video_path": video_path,
                "label": _parse_label(row[label_key]),
            })

        return normalized

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        imgs, _ = VideoCapture.load_frames_from_video(
            record["video_path"],
            self.num_frames,
            self.config.num_prompts,
            self.video_sample_type,
        )

        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            "video_id": record["video_id"],
            "video": imgs,
            "label": record["label"],
        }
