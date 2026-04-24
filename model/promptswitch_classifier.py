import torch
from torch import nn
from transformers.models.clip import CLIPConfig

from model.prompt_clip import CLIPModel


class PromptSwitchClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        clip_config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        clip_config.vision_config.update({
            "num_frames": self.config.num_frames,
            "num_prompts": self.config.num_prompts,
        })
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", config=clip_config)

        feature_dim = clip_config.projection_dim
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(feature_dim, config.num_classes)

        self.clip_params = []
        self.noclip_params = []

        self._freeze_unused_text_branch()
        if config.freeze_vision_backbone:
            self._freeze_visual_backbone_except_prompt_modules()

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("clip.vision_model.encoder.prompt_embedding") or \
               name.startswith("clip.vision_model.attn") or \
               name.startswith("classifier") or \
               name.startswith("dropout"):
                self.noclip_params.append(param)
            elif name.startswith("clip.vision_model") or name.startswith("clip.visual_projection"):
                self.clip_params.append(param)

        nn.init.normal_(
            self.clip.vision_model.encoder.prompt_embedding,
            mean=0.0,
            std=clip_config.vision_config.hidden_size ** -0.5 * clip_config.vision_config.initializer_range,
        )
        nn.init.zeros_(self.clip.vision_model.attn.out_proj.weight.data)
        nn.init.zeros_(self.clip.vision_model.attn.out_proj.bias.data)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def _freeze_unused_text_branch(self):
        for name, param in self.clip.named_parameters():
            if name.startswith("text_model") or name.startswith("text_projection") or name.startswith("logit_scale"):
                param.requires_grad = False

    def _freeze_visual_backbone_except_prompt_modules(self):
        for name, param in self.clip.named_parameters():
            if name.startswith("vision_model.encoder.prompt_embedding") or name.startswith("vision_model.attn"):
                continue
            if name.startswith("vision_model") or name.startswith("visual_projection"):
                param.requires_grad = False

    def _set_video_frame_count(self, num_frames):
        vision_model = self.clip.vision_model
        vision_model.nf = num_frames
        vision_model.config.num_frames = num_frames
        vision_model.encoder.nf = num_frames
        vision_model.encoder.config.num_frames = num_frames
        for layer in vision_model.encoder.layers:
            layer.nf = num_frames

    def _pool_video_features(self, video_features):
        if self.config.temporal_pooling == "avg":
            return video_features.mean(dim=1)
        if self.config.temporal_pooling == "max":
            return video_features.max(dim=1).values
        raise ValueError(f"Unsupported temporal pooling: {self.config.temporal_pooling}")

    def forward(self, data):
        bs = data["video"].shape[0]
        video_data = data["video"]
        num_frames = video_data.shape[1]

        self._set_video_frame_count(num_frames)
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        frame_features = self.clip.get_image_features(video_data)
        frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)
        frame_features = frame_features.reshape(bs, -1, frame_features.size(-1))

        video_repr = self._pool_video_features(frame_features)
        logits = self.classifier(self.dropout(video_repr))

        return {
            "logits": logits,
            "video_features": frame_features,
            "video_repr": video_repr,
        }
