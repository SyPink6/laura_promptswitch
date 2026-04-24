import argparse
import csv
import os
import random
import re
from collections import defaultdict

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Split a frame-level CSV into train/val manifests by grouped video id.")
    parser.add_argument("--input_csv", type=str, required=True, help="Source CSV with img_path and label columns")
    parser.add_argument("--train_out", type=str, required=True, help="Output CSV path for train split")
    parser.add_argument("--val_out", type=str, required=True, help="Output CSV path for val split")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio by grouped videos")
    parser.add_argument("--seed", type=int, default=24, help="Random seed")
    return parser.parse_args()


def split_frame_stem(stem):
    match = re.match(r"^(.*)_(\d+)$", stem)
    if match is None:
        return stem
    return match.group(1)


def build_video_id(row):
    frame_path = row["img_path"]
    stem = os.path.splitext(os.path.basename(frame_path))[0]
    return split_frame_stem(stem)


def write_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if "img_path" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain at least 'img_path' and 'label' columns.")

    rows = df.to_dict("records")
    grouped = defaultdict(list)
    label_by_video = {}
    for row in rows:
        video_id = build_video_id(row)
        label = row["label"]
        if video_id in label_by_video and label_by_video[video_id] != label:
            raise ValueError(f"Inconsistent labels found for grouped video id '{video_id}'.")
        label_by_video[video_id] = label
        grouped[label].append(video_id)

    rng = random.Random(args.seed)
    train_video_ids = set()
    val_video_ids = set()

    for label, video_ids in grouped.items():
        unique_ids = sorted(set(video_ids))
        rng.shuffle(unique_ids)
        num_val = max(1, int(round(len(unique_ids) * args.val_ratio))) if len(unique_ids) > 1 else 0
        val_ids = set(unique_ids[:num_val])
        train_ids = set(unique_ids[num_val:]) or val_ids
        if not train_ids:
            raise ValueError(f"Label '{label}' has no train videos after split.")
        train_video_ids.update(train_ids)
        val_video_ids.update(val_ids)

    train_rows = []
    val_rows = []
    for row in rows:
        video_id = build_video_id(row)
        if video_id in val_video_ids:
            val_rows.append(row)
        else:
            train_rows.append(row)

    fieldnames = list(df.columns)
    write_rows(args.train_out, train_rows, fieldnames)
    write_rows(args.val_out, val_rows, fieldnames)

    print(f"Input rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Train videos: {len(train_video_ids)}")
    print(f"Val videos: {len(val_video_ids)}")


if __name__ == "__main__":
    main()
