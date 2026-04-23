import torch
import torch.nn as nn


class FaceToTextProjector(nn.Module):
    """
    Maps a 512-d TransFace embedding into a 512-d CLIP text pseudo-word embedding.

    For stage-1 prototyping we default to an identity-initialized linear layer so the
    whole pipeline can run before introducing a learned projector.
    """

    def __init__(self, in_dim=512, out_dim=512, mode="identity"):
        super().__init__()
        self.mode = mode

        if mode == "identity":
            self.net = nn.Linear(in_dim, out_dim, bias=True)
            if in_dim == out_dim:
                nn.init.eye_(self.net.weight)
            else:
                nn.init.xavier_uniform_(self.net.weight)
            nn.init.zeros_(self.net.bias)

        elif mode == "linear":
            self.net = nn.Linear(in_dim, out_dim, bias=True)
            nn.init.xavier_uniform_(self.net.weight)
            nn.init.zeros_(self.net.bias)

        elif mode == "mlp":
            hidden_dim = out_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, out_dim),
            )
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        else:
            raise ValueError(f"Unsupported projector mode: {mode}")

    def forward(self, face_embedding):
        return self.net(face_embedding)
