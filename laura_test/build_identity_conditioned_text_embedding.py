import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from extract_transface_embedding import (
    build_model as build_transface_model,
    extract_embedding as extract_face_embedding,
    load_image,
    load_official_backbone_builder,
    maybe_align_face,
    preprocess_image,
)
from face_to_text_projector import FaceToTextProjector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-1 prototype: build identity-conditioned real/fake text embeddings."
    )
    parser.add_argument("--image", type=str, default="laura_test/000.png")
    parser.add_argument(
        "--transface-repo",
        type=str,
        required=True,
        help="Path to a local clone of the official TransFace repository.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="laura_added/glint360k_model_TransFace_L.pt",
        help="Path to the TransFace checkpoint.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="vit_l_dp005_mask_005",
        help="Official TransFace backbone name.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP text encoder checkpoint.",
    )
    parser.add_argument(
        "--placeholder-word",
        type=str,
        default="this",
        help="The token whose embedding will be replaced by the pseudo-word.",
    )
    parser.add_argument(
        "--real-template",
        type=str,
        default="a real video of a this person",
        help="Real-class prompt template.",
    )
    parser.add_argument(
        "--fake-template",
        type=str,
        default="a fake video of a this person",
        help="Fake-class prompt template.",
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
        help="Treat the image as an already aligned face crop.",
    )
    parser.add_argument(
        "--projector-mode",
        type=str,
        default="identity",
        choices=["identity", "linear", "mlp"],
        help="Stage-1 projector type. 'identity' is the safest starting point.",
    )
    parser.add_argument(
        "--projector-ckpt",
        type=str,
        default=None,
        help="Optional checkpoint for a pretrained face-to-text projector.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="laura_test/identity_conditioned_text_embeddings.json",
        help="Where to save metadata and prompt previews.",
    )
    parser.add_argument(
        "--save-npy",
        type=str,
        default="laura_test/identity_conditioned_text_embeddings.npy",
        help="Where to save the real/fake text embeddings.",
    )
    return parser.parse_args()


def get_runtime_device(device_str: str):
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_face_embedding(args):
    get_model = load_official_backbone_builder(args.transface_repo)
    transface_model = build_transface_model(get_model, args.network)
    image = load_image(args.image)
    aligned = maybe_align_face(image, aligned=args.aligned)
    image_tensor = preprocess_image(aligned)
    raw_embedding, norm_embedding, actual_device = extract_face_embedding(
        model=transface_model,
        image_tensor=image_tensor,
        weights_path=args.weights,
        device=args.device,
    )
    return raw_embedding, norm_embedding, actual_device, aligned


def load_clip_text_stack(model_name: str, device: torch.device):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def find_placeholder_positions(tokenizer, input_ids: torch.Tensor, placeholder_word: str):
    placeholder_ids = tokenizer.encode(placeholder_word, add_special_tokens=False)
    if len(placeholder_ids) != 1:
        raise RuntimeError(
            f"Placeholder word '{placeholder_word}' expands to {len(placeholder_ids)} tokens. "
            "Stage-1 expects a single token placeholder."
        )

    placeholder_id = placeholder_ids[0]
    positions = []
    for row in input_ids:
        hits = (row == placeholder_id).nonzero(as_tuple=False).flatten()
        if len(hits) == 0:
            raise RuntimeError(
                f"Could not find placeholder word '{placeholder_word}' in tokenized prompt."
            )
        positions.append(int(hits[0].item()))
    return positions, placeholder_id


def build_custom_text_features(clip_model, input_ids, attention_mask, replacement_positions, pseudo_word):
    text_model = clip_model.text_model
    bsz, seq_len = input_ids.shape
    device = input_ids.device

    position_ids = text_model.embeddings.position_ids[:, :seq_len]
    token_embeds = text_model.embeddings.token_embedding(input_ids)

    for batch_idx, pos in enumerate(replacement_positions):
        token_embeds[batch_idx, pos, :] = pseudo_word[batch_idx]

    position_embeds = text_model.embeddings.position_embedding(position_ids)
    hidden_states = token_embeds + position_embeds

    causal_attention_mask = torch.empty(bsz, seq_len, seq_len, device=device)
    causal_attention_mask.fill_(float("-inf"))
    causal_attention_mask.triu_(1)
    causal_attention_mask = causal_attention_mask.unsqueeze(1)

    expanded_attention_mask = None
    if attention_mask is not None:
        expanded_attention_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)
        expanded_attention_mask = (1.0 - expanded_attention_mask) * torch.finfo(hidden_states.dtype).min

    encoder_outputs = text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=expanded_attention_mask,
        causal_attention_mask=causal_attention_mask,
        return_dict=False,
    )
    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_model.final_layer_norm(last_hidden_state)
    pooled_output = last_hidden_state[torch.arange(bsz, device=device), input_ids.argmax(dim=-1)]

    text_features = clip_model.text_projection(pooled_output)
    text_features = F.normalize(text_features, dim=-1)
    return text_features, token_embeds


def maybe_load_projector(projector, projector_ckpt: str):
    if not projector_ckpt:
        return projector

    checkpoint = torch.load(projector_ckpt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    projector.load_state_dict(state_dict, strict=True)
    return projector


def main():
    args = parse_args()
    device = get_runtime_device(args.device)

    raw_face_embedding, norm_face_embedding, transface_device, aligned_face = build_face_embedding(args)

    tokenizer, clip_model = load_clip_text_stack(args.clip_model, device)

    projector = FaceToTextProjector(
        in_dim=512,
        out_dim=clip_model.text_projection.out_features,
        mode=args.projector_mode,
    )
    projector = maybe_load_projector(projector, args.projector_ckpt).to(device)
    projector.eval()

    prompts = [args.real_template, args.fake_template]
    text_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    replacement_positions, placeholder_id = find_placeholder_positions(
        tokenizer=tokenizer,
        input_ids=text_inputs["input_ids"],
        placeholder_word=args.placeholder_word,
    )

    face_tensor = torch.from_numpy(norm_face_embedding).float().unsqueeze(0).to(device)
    pseudo_word_single = projector(face_tensor)
    pseudo_word = pseudo_word_single.repeat(len(prompts), 1)

    text_features, token_embeds = build_custom_text_features(
        clip_model=clip_model,
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs.get("attention_mask"),
        replacement_positions=replacement_positions,
        pseudo_word=pseudo_word,
    )

    real_text = text_features[0].detach().cpu().numpy()
    fake_text = text_features[1].detach().cpu().numpy()
    stacked = np.stack([real_text, fake_text], axis=0).astype(np.float32)
    np.save(args.save_npy, stacked)

    cosine_rf = float(np.dot(real_text, fake_text))
    preview = {
        "image": str(Path(args.image).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "transface_repo": str(Path(args.transface_repo).resolve()),
        "clip_model": args.clip_model,
        "transface_device": transface_device,
        "clip_device": str(device),
        "aligned_input_shape": list(aligned_face.shape),
        "placeholder_word": args.placeholder_word,
        "placeholder_token_id": int(placeholder_id),
        "replacement_positions": replacement_positions,
        "projector_mode": args.projector_mode,
        "projector_checkpoint": str(Path(args.projector_ckpt).resolve()) if args.projector_ckpt else None,
        "real_template": args.real_template,
        "fake_template": args.fake_template,
        "face_embedding_dim": int(norm_face_embedding.shape[0]),
        "pseudo_word_dim": int(pseudo_word_single.shape[-1]),
        "text_embedding_dim": int(real_text.shape[0]),
        "real_fake_cosine_similarity": cosine_rf,
        "face_embedding_preview": norm_face_embedding[:8].tolist(),
        "pseudo_word_preview": pseudo_word_single[0].detach().cpu().numpy()[:8].tolist(),
        "real_text_preview": real_text[:8].tolist(),
        "fake_text_preview": fake_text[:8].tolist(),
        "save_npy": str(Path(args.save_npy).resolve()),
    }

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)

    print("Stage-1 identity-conditioned text embedding finished.")
    print(f"Face embedding dim: {norm_face_embedding.shape[0]}")
    print(f"Pseudo-word dim: {pseudo_word_single.shape[-1]}")
    print(f"Text embedding dim: {real_text.shape[0]}")
    print(f"Real/Fake cosine similarity: {cosine_rf:.6f}")
    print(f"Saved text embeddings to: {args.save_npy}")
    print(f"Saved metadata to: {args.save_json}")
    print("Real prompt first 8 dims:")
    print(real_text[:8])
    print("Fake prompt first 8 dims:")
    print(fake_text[:8])


if __name__ == "__main__":
    main()
