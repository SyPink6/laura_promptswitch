import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
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

    def _read_rows(self, manifest_path):
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
        return rows

    def _resolve_path(self, path_value):
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(self.videos_dir, path_value)

    def _split_frame_stem(self, stem):
        match = re.match(r"^(.*)_(\d+)$", stem)
        if match is None:
            return stem, -1
        return match.group(1), int(match.group(2))

    def _normalize_video_manifest(self, rows):
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

            video_path = self._resolve_path(row[video_key])
            record_id = row.get("video_id", os.path.splitext(os.path.basename(video_path))[0])
            normalized.append({
                "video_id": str(record_id),
                "video_path": video_path,
                "label": _parse_label(row[label_key]),
                "record_type": "video",
            })
        return normalized

    def _normalize_frame_manifest(self, rows):
        grouped = {}
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {idx} is not a dictionary.")

            frame_key = next(
                (key for key in ["img_path", "image_path", "frame", "frame_path", "img", "image"] if key in row),
                None,
            )
            label_key = next(
                (key for key in ["label", "target", "class", "y"] if key in row),
                None,
            )
            if frame_key is None or label_key is None:
                raise ValueError(
                    f"Frame manifest row {idx} must contain an image path field and a label field. "
                    "Accepted frame keys: img_path/image_path/frame/frame_path/img/image; "
                    "label keys: label/target/class/y."
                )

            frame_path = self._resolve_path(row[frame_key])
            stem = os.path.splitext(os.path.basename(frame_path))[0]
            default_video_id, frame_index = self._split_frame_stem(stem)
            video_id = str(row.get("video_id", default_video_id))
            label = _parse_label(row[label_key])

            entry = grouped.setdefault(video_id, {"video_id": video_id, "label": label, "frame_paths": []})
            if entry["label"] != label:
                raise ValueError(f"Inconsistent labels found for grouped video_id '{video_id}'.")
            entry["frame_paths"].append((frame_index, frame_path))

        normalized = []
        for video_id, entry in grouped.items():
            frame_paths = [path for _, path in sorted(entry["frame_paths"], key=lambda item: item[0])]
            normalized.append({
                "video_id": video_id,
                "frame_paths": frame_paths,
                "label": entry["label"],
                "record_type": "frames",
            })

        if not normalized:
            raise ValueError("No grouped frame records were built from the frame manifest.")
        return normalized

    def _load_manifest(self, manifest_path):
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: '{manifest_path}'")

        rows = self._read_rows(manifest_path)
        first_row = rows[0]
        if any(key in first_row for key in ["img_path", "image_path", "frame", "frame_path", "img", "image"]):
            return self._normalize_frame_manifest(rows)
        return self._normalize_video_manifest(rows)

    def _sample_frame_indices(self, num_available_frames):
        acc_samples = min(self.config.num_prompts, num_available_frames)
        intervals = torch.linspace(0, num_available_frames, steps=acc_samples + 1, dtype=torch.int64).tolist()
        ranges = [(int(intervals[idx]), int(intervals[idx + 1]) - 1) for idx in range(acc_samples)]

        num_chunks = self.num_frames // self.config.num_prompts if self.config.num_prompts > 0 else 1
        frame_indices = []
        if self.video_sample_type == "rand":
            for _ in range(num_chunks):
                for start, end in ranges:
                    if end < start:
                        frame_indices.append(start)
                    else:
                        frame_indices.append(random.randint(start, end))
        else:
            for chunk_idx in range(num_chunks):
                for start, end in ranges:
                    if end < start:
                        frame_indices.append(start)
                    else:
                        span = end - start + 1
                        offset = max(span // (num_chunks + 1) * (chunk_idx + 1) - 1, 0)
                        frame_indices.append(min(start + offset, end))

        if not frame_indices:
            frame_indices = [0]
        return frame_indices

    def _load_frames_from_paths(self, frame_paths):
        if not frame_paths:
            raise ValueError("Grouped frame record has no frame paths.")

        frame_indices = self._sample_frame_indices(len(frame_paths))
        frames = []
        for frame_index in frame_indices:
            frame_path = frame_paths[min(frame_index, len(frame_paths) - 1)]
            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                frame = torch.from_numpy(np.array(image))
            frame = frame.permute(2, 0, 1)
            frames.append(frame)

        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        return torch.stack(frames).float() / 255.0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if record["record_type"] == "video":
            imgs, _ = VideoCapture.load_frames_from_video(
                record["video_path"],
                self.num_frames,
                self.config.num_prompts,
                self.video_sample_type,
            )
        else:
            imgs = self._load_frames_from_paths(record["frame_paths"])

        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            "video_id": record["video_id"],
            "video": imgs,
            "label": record["label"],
        }
