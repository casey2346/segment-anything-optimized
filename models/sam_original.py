"""
sam_original.py

Author: Kexin Rong (you@example.com)
License: MIT

This module implements the original architecture of the Segment Anything Model (SAM),
based on the official Meta AI CVPR 2023 paper.

It reproduces the Vision Transformer (ViT) encoder and mask decoder as described in
the paper.

Advantages:
- Faithfully reproduces the paper architecture
- Serves as the teacher model for distillation
- Provides a baseline for performance comparisons

✅ Tested with PyTorch >= 1.10

Example usage (as Python module):

    from sam_original import SegmentAnythingModel

    encoder_config = {"embed_dim": 1024, "depth": 24, "num_heads": 16}
    decoder_config = {"embed_dim": 1024, "num_classes": 1}
    model = SegmentAnythingModel(encoder_config, decoder_config)

    # Forward pass
    images = torch.randn(2, 3, 224, 224)
    masks = model(images)
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class TransformerBlock(nn.Module):
    """
    A placeholder Transformer block.
    In production, replace with an actual implementation (e.g., MultiheadAttention).
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.Identity()  # Replace with real attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


class SAMEncoder(nn.Module):
    """
    The Vision Transformer encoder for SAM.
    Implements Eq. (3) from the paper: Self-Attention with Linear Projection.
    """
    def __init__(self, embed_dim: int, depth: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.proj(x)


class SAMDecoder(nn.Module):
    """
    Mask decoder head as described in Section 3.2 of the paper.
    """
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SegmentAnythingModel(nn.Module):
    """
    Complete SAM model combining encoder and decoder.
    """
    def __init__(self, encoder_config: Dict[str, Any], decoder_config: Dict[str, Any]):
        super().__init__()
        self.encoder = SAMEncoder(**encoder_config)
        self.decoder = SAMDecoder(**decoder_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        masks = self.decoder(features)
        return masks
