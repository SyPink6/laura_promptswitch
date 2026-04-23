import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a face embedding with TransFace without touching PromptSwitch."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="laura_test/000.png",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="laura_added/glint360k_model_TransFace_L.pt",
        help="Path to the TransFace checkpoint.",
    )
    parser.add_argument(
        "--transface-repo",
        type=str,
        default=os.environ.get("TRANSFACE_REPO"),
        help=(
            "Path to a local clone of the official TransFace repository. "
            "The script imports backbones.get_model from there."
        ),
    )
    parser.add_argument(
        "--network",
        type=str,
        default="vit_l_dp005_mask_005",
        help="Official TransFace backbone name for the checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--aligned",
        action="store_true",
        help="Treat the input image as an already aligned face crop.",
    )
    parser.add_argument(
        "--save-npy",
        type=str,
        default="laura_test/transface_embedding.npy",
        help="Where to save the L2-normalized embedding as .npy.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="laura_test/transface_embedding.json",
        help="Where to save metadata and a preview of the embedding.",
    )
    return parser.parse_args()


def add_transface_repo_to_path(repo_path: str):
    if not repo_path:
        raise RuntimeError(
            "Missing --transface-repo. Clone the official repo first, for example:\n"
            "  git clone https://github.com/DanJun6737/TransFace.git D:/path/to/TransFace"
        )

    repo = Path(repo_path).expanduser().resolve()
    backbones_init = repo / "backbones" / "__init__.py"
    if not backbones_init.exists():
        raise FileNotFoundError(
            f"Could not find backbones/__init__.py under '{repo}'. "
            "Make sure --transface-repo points to a local clone of the official repository."
        )

    sys.path.insert(0, str(repo))
    return repo


def load_official_backbone_builder(repo_path: str):
    add_transface_repo_to_path(repo_path)
    from backbones import get_model  # type: ignore

    return get_model


def load_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def maybe_align_face(image_bgr: np.ndarray, aligned: bool):
    if aligned:
        return cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)

    h, w = image_bgr.shape[:2]
    if h == 112 and w == 112:
        return image_bgr

    raise RuntimeError(
        "The image is not marked as aligned and is not already 112x112.\n"
        "TransFace expects an aligned face crop. Either:\n"
        "1. preprocess the face with a detector/alignment tool first, or\n"
        "2. rerun this script with --aligned if your image is already a face crop."
    )


def preprocess_image(aligned_bgr: np.ndarray):
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    image = rgb.astype(np.float32)
    # Official TransFace normalization in torch2onnx.py:
    # img = (img / 255. - 0.5) / 0.5
    image = (image / 255.0 - 0.5) / 0.5
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image


def build_model(get_model, network: str):
    # Official TransFace torch2onnx.py uses:
    # get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    return get_model(network, dropout=0.0, fp16=False, num_features=512)


def load_state_dict_safely(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "backbone"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def extract_embedding(model, image_tensor, weights_path: str, device: str):
    import torch
    import torch.nn.functional as F

    actual_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = load_state_dict_safely(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(actual_device)
    model.eval()

    batch = torch.from_numpy(image_tensor).to(actual_device)

    with torch.no_grad():
        embedding = model(batch)
        embedding = embedding.float()
        embedding_norm = F.normalize(embedding, dim=1)

    return embedding.cpu().numpy()[0], embedding_norm.cpu().numpy()[0], str(actual_device)


def save_outputs(raw_embedding, norm_embedding, save_npy: str, save_json: str, meta: dict):
    np.save(save_npy, norm_embedding.astype(np.float32))

    payload = dict(meta)
    payload["raw_embedding_dim"] = int(raw_embedding.shape[0])
    payload["normalized_embedding_dim"] = int(norm_embedding.shape[0])
    payload["raw_embedding_preview"] = raw_embedding[:8].tolist()
    payload["normalized_embedding_preview"] = norm_embedding[:8].tolist()

    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    get_model = load_official_backbone_builder(args.transface_repo)
    model = build_model(get_model, args.network)

    image = load_image(args.image)
    aligned = maybe_align_face(image, aligned=args.aligned)
    image_tensor = preprocess_image(aligned)
    raw_embedding, norm_embedding, actual_device = extract_embedding(
        model=model,
        image_tensor=image_tensor,
        weights_path=args.weights,
        device=args.device,
    )

    meta = {
        "image": str(Path(args.image).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "transface_repo": str(Path(args.transface_repo).resolve()) if args.transface_repo else None,
        "network": args.network,
        "device": actual_device,
        "aligned_input_shape": list(aligned.shape),
        "save_npy": str(Path(args.save_npy).resolve()),
        "save_json": str(Path(args.save_json).resolve()),
    }
    save_outputs(raw_embedding, norm_embedding, args.save_npy, args.save_json, meta)

    print("TransFace embedding extraction finished.")
    print(f"Device: {actual_device}")
    print(f"Aligned face shape: {aligned.shape}")
    print(f"Embedding dim: {norm_embedding.shape[0]}")
    print(f"Saved normalized embedding to: {args.save_npy}")
    print(f"Saved metadata preview to: {args.save_json}")
    print("First 8 dims of normalized embedding:")
    print(norm_embedding[:8])


if __name__ == "__main__":
    main()
