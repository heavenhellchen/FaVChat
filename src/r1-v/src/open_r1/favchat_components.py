from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FaVChatState:
    facial_tokens: torch.Tensor
    general_tokens: torch.Tensor
    fused_tokens: torch.Tensor
    text_tokens: torch.Tensor
    general_weight: torch.Tensor
    facial_weight: torch.Tensor


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_query = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        residual = query
        query = self.norm_query(query)
        key_value = self.norm_kv(key_value)
        attended, _ = self.attn(query=query, key=key_value, value=key_value, need_weights=False)
        hidden = residual + attended
        hidden = hidden + self.mlp(hidden)
        return hidden


class LowLevelPromptQueryAggregator(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_size=hidden_size, num_heads=num_heads) for _ in range(num_layers)]
        )

    def forward(self, text_tokens: torch.Tensor, facial_feature_pyramid: Iterable[torch.Tensor]) -> torch.Tensor:
        fused = text_tokens
        for layer, visual_tokens in zip(self.layers, facial_feature_pyramid):
            fused = layer(fused, visual_tokens)
        return fused


class PromptConditionedQFormer(nn.Module):
    def __init__(self, hidden_size: int, num_queries: int, num_heads: int, num_layers: int):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        self.prompt_adapter = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_size=hidden_size, num_heads=num_heads) for _ in range(num_layers)]
        )

    def forward(self, text_tokens: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        pooled_prompt = text_tokens.mean(dim=1, keepdim=True)
        prompt_bias = self.prompt_adapter(pooled_prompt)
        queries = self.query_tokens.expand(text_tokens.size(0), -1, -1) + prompt_bias
        for layer in self.layers:
            queries = layer(queries, visual_tokens)
        return queries


class WeightAdapter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.visual_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        visual_summary = visual_tokens.mean(dim=1)
        text_summary = text_tokens.mean(dim=1)
        hidden = F.gelu(self.visual_proj(visual_summary) + self.text_proj(text_summary))
        return self.score(hidden).squeeze(-1)


class FaVChatPromptQueryEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_low_level_layers: int = 4,
        num_qformer_layers: int = 2,
        num_queries: int = 16,
    ):
        super().__init__()
        self.low_level = LowLevelPromptQueryAggregator(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_low_level_layers,
        )
        self.general_qformer = PromptConditionedQFormer(
            hidden_size=hidden_size,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_qformer_layers,
        )
        self.facial_qformer = PromptConditionedQFormer(
            hidden_size=hidden_size,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_qformer_layers,
        )
        self.general_weight_adapter = WeightAdapter(hidden_size)
        self.facial_weight_adapter = WeightAdapter(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_tokens: torch.Tensor,
        general_tokens: torch.Tensor,
        facial_feature_pyramid: Iterable[torch.Tensor],
    ) -> FaVChatState:
        low_level_facial = self.low_level(text_tokens=text_tokens, facial_feature_pyramid=facial_feature_pyramid)
        general_tokens = self.general_qformer(text_tokens=text_tokens, visual_tokens=general_tokens)
        facial_tokens = self.facial_qformer(text_tokens=text_tokens, visual_tokens=low_level_facial)

        general_logits = self.general_weight_adapter(general_tokens, text_tokens)
        facial_logits = self.facial_weight_adapter(facial_tokens, text_tokens)
        weights = torch.softmax(torch.stack([general_logits, facial_logits], dim=-1), dim=-1)
        general_weight = weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        facial_weight = weights[:, 1].unsqueeze(-1).unsqueeze(-1)

        fused_tokens = self.output_norm(general_tokens * general_weight + facial_tokens * facial_weight)
        return FaVChatState(
            facial_tokens=facial_tokens,
            general_tokens=general_tokens,
            fused_tokens=fused_tokens,
            text_tokens=text_tokens,
            general_weight=general_weight.squeeze(-1),
            facial_weight=facial_weight.squeeze(-1),
        )


class SampleValueBaseline(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, sample_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(sample_embedding).squeeze(-1)


def build_feature_pyramid(hidden_states: tuple[torch.Tensor, ...], num_layers: int) -> list[torch.Tensor]:
    if len(hidden_states) <= 1:
        return [hidden_states[-1]] * num_layers
    selected = torch.linspace(1, len(hidden_states) - 1, steps=num_layers).round().long().tolist()
    return [hidden_states[idx] for idx in selected]


def pairwise_reward_statistics(rewards: torch.Tensor, num_generations: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped = rewards.view(-1, num_generations)
    reward_diffs = []
    absolute_diffs = []
    pair_counts = []
    for sample_rewards in grouped:
        diffs = []
        for winner in range(num_generations):
            for loser in range(winner + 1, num_generations):
                delta = (sample_rewards[winner] - sample_rewards[loser]).abs()
                diffs.append(delta)
        if diffs:
            diffs_tensor = torch.stack(diffs)
            reward_diffs.append(diffs_tensor.mean())
            absolute_diffs.append(torch.exp(torch.log(diffs_tensor.abs() + 1e-6).mean()))
            pair_counts.append(diffs_tensor.numel())
        else:
            zero = sample_rewards.new_tensor(0.0)
            reward_diffs.append(zero)
            absolute_diffs.append(zero)
            pair_counts.append(0)
    return (
        torch.stack(reward_diffs),
        torch.stack(absolute_diffs),
        torch.tensor(pair_counts, device=rewards.device, dtype=rewards.dtype),
    )
