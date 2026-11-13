"""Expert components from MATH_SPEC §2; full-model registration belongs to #18.

``Experts(batch)`` returns ``{name: (h_k, z_k)}`` for img, txt and int, with
shapes [B, h] and [B, num_outputs]. Binary logits retain their final size-1 axis.
Text masks use True for valid tokens and must retain at least one token per row.
"""

import math

import torch
from torch import nn


def _mlp(in_features: int, hidden: int, out_features: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, out_features))


class InputProjection(nn.Sequential):
    """Shared token/pooled map: d → h, layer normalization, dropout."""

    def __init__(self, d: int, h: int, dropout: float):
        super().__init__(nn.Linear(d, h, bias=False), nn.LayerNorm(h), nn.Dropout(dropout))


class AttentionPool(nn.Module):
    """D-017 reassociation: score with W_kᵀq, then apply W_o after pooling."""

    def __init__(self, h: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(h) / math.sqrt(h))
        self.key = nn.Linear(h, h, bias=False)
        self.output = nn.Linear(h, h, bias=False)

    def attention_weights(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        u = self.key.weight.T @ self.query
        scores = (x @ u) / math.sqrt(x.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask, -torch.inf)
        return scores.softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        weights = self.attention_weights(x, mask)
        return self.output((weights.unsqueeze(-1) * x).sum(dim=1))


class CrossAttentionLayer(nn.Module):
    """Pre-LN bidirectional cross-attention with simultaneous modality updates."""

    def __init__(self, h: int, heads: int, dropout: float):
        super().__init__()
        self.norm_v = nn.LayerNorm(h)
        self.norm_t = nn.LayerNorm(h)
        self.attn_v = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.attn_t = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.ffn_norm_v = nn.LayerNorm(h)
        self.ffn_norm_t = nn.LayerNorm(h)
        self.ffn_v = _mlp(h, 2 * h, h, dropout)
        self.ffn_t = _mlp(h, 2 * h, h, dropout)

    def forward(self, image: torch.Tensor, text: torch.Tensor,
                text_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v, t = self.norm_v(image), self.norm_t(text)
        image = image + self.attn_v(v, t, t, key_padding_mask=~text_mask,
                                    need_weights=False)[0]
        text = text + self.attn_t(t, v, v, need_weights=False)[0]
        return (image + self.ffn_v(self.ffn_norm_v(image)),
                text + self.ffn_t(self.ffn_norm_t(text)))


class UnimodalExpert(nn.Module):
    """Image or text expert operating on the shared, already projected features."""

    def __init__(self, h: int, dropout: float, num_outputs: int):
        super().__init__()
        self.pool = AttentionPool(h)
        self.mlp = _mlp(2 * h, h, h, dropout)
        self.head = nn.Linear(h, num_outputs)

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor,
                mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.mlp(torch.cat((self.pool(tokens, mask), pooled), dim=-1))
        return hidden, self.head(hidden)


class InteractionExpert(nn.Module):
    """Cross-attend projected tokens and cross-fuse raw d-dimensional pooled vectors."""

    def __init__(self, d: int, h: int, heads: int, n_cross_layers: int,
                 cross_fusion_dim: int, dropout: float, num_outputs: int):
        super().__init__()
        self.layers = nn.ModuleList(CrossAttentionLayer(h, heads, dropout)
                                    for _ in range(n_cross_layers))
        self.pool_v = AttentionPool(h)
        self.pool_t = AttentionPool(h)
        self.cross_v = nn.Linear(d, cross_fusion_dim, bias=False)
        self.cross_t = nn.Linear(d, cross_fusion_dim, bias=False)
        self.mlp = _mlp(2 * h + cross_fusion_dim ** 2, h, h, dropout)
        self.head = nn.Linear(h, num_outputs)

    def forward(self, image: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor,
                image_pooled: torch.Tensor,
                text_pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            image, text = layer(image, text, text_mask)
        pv, pt = self.cross_v(image_pooled), self.cross_t(text_pooled)
        cross = (pv.unsqueeze(-1) * pt.unsqueeze(-2)).flatten(start_dim=1)
        hidden = self.mlp(torch.cat((self.pool_v(image, None),
                                     self.pool_t(text, text_mask), cross), dim=-1))
        return hidden, self.head(hidden)


class Experts(nn.Module):
    """Three experts with single shared φ_v/φ_t maps, ready for #18's builder.

    All architectural arguments are explicit. Each token projection is computed
    once so unimodal and interaction experts also share its dropout realization.
    """

    expert_names = ("img", "txt", "int")

    def __init__(self, d: int, h: int, heads: int, n_cross_layers: int,
                 cross_fusion_dim: int, dropout: float, num_outputs: int):
        super().__init__()
        self.phi_v = InputProjection(d, h, dropout)
        self.phi_t = InputProjection(d, h, dropout)
        self.img = UnimodalExpert(h, dropout, num_outputs)
        self.txt = UnimodalExpert(h, dropout, num_outputs)
        self.interaction = InteractionExpert(d, h, heads, n_cross_layers,
                                             cross_fusion_dim, dropout, num_outputs)

    def forward(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        mask = batch["text_mask"]
        image = self.phi_v(batch["image_tokens"])
        text = self.phi_t(batch["text_tokens"].masked_fill(~mask.unsqueeze(-1), 0))
        return {
            "img": self.img(image, self.phi_v(batch["image_pooled"]), None),
            "txt": self.txt(text, self.phi_t(batch["text_pooled"]), mask),
            "int": self.interaction(image, text, mask,
                                    batch["image_pooled"], batch["text_pooled"]),
        }
