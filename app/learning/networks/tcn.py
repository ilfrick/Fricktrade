# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Temporal Convolutional Network (TCN) feature extractor for RL policies.

TCNs use dilated causal convolutions to capture temporal dependencies
efficiently, making them well-suited for financial time series.
"""

from __future__ import annotations

import logging
from typing import Callable

import gymnasium as gym
import numpy as np

try:
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    BaseFeaturesExtractor = object
    nn = None
    torch = None


logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module if TORCH_AVAILABLE else object):
    """
    A single temporal block with dilated causal convolution.

    Structure:
        Conv1d -> BatchNorm -> ReLU -> Dropout ->
        Conv1d -> BatchNorm -> ReLU -> Dropout ->
        Residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (1x1 conv if channels differ)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        out = self.conv1(x)
        # Causal: remove future timesteps introduced by padding
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = self.residual(x)
        # Match temporal dimension
        if res.size(2) != out.size(2):
            res = res[:, :, -out.size(2):]

        return self.relu_out(out + res)


class TCN(nn.Module if TORCH_AVAILABLE else object):
    """
    Temporal Convolutional Network.

    Uses exponentially increasing dilations to capture long-range dependencies
    while maintaining causal structure.
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_channels is None:
            num_channels = [64, 64, 64]

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features) or (batch, seq_len, features)
        if x.dim() == 2:
            # Add sequence dimension (treat as single timestep)
            x = x.unsqueeze(2)  # (batch, features, 1)
        elif x.dim() == 3:
            # Transpose to (batch, features, seq_len) for Conv1d
            x = x.transpose(1, 2)

        out = self.network(x)

        # Global average pooling over time dimension
        out = out.mean(dim=2)  # (batch, channels)

        return out


class TCNExtractor(BaseFeaturesExtractor):
    """
    TCN-based feature extractor for Stable Baselines3 PPO.

    This replaces the default MLP feature extractor with a TCN
    that better captures temporal patterns in the observation.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TCN feature extractor")

        super().__init__(observation_space, features_dim)

        input_dim = int(np.prod(observation_space.shape))
        num_channels = [features_dim] * num_layers

        self.tcn = TCN(
            input_dim=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Final projection to features_dim
        self.fc = nn.Linear(self.tcn.output_dim, features_dim)

        logger.info(
            "Created TCN extractor: input_dim=%d, features_dim=%d, layers=%d",
            input_dim,
            features_dim,
            num_layers,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        tcn_out = self.tcn(observations)
        return self.fc(tcn_out)


def create_tcn_policy_kwargs(
    features_dim: int = 64,
    num_layers: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.1,
    net_arch: list | None = None,
) -> dict:
    """
    Create policy_kwargs dict for PPO with TCN feature extractor.

    Usage:
        policy_kwargs = create_tcn_policy_kwargs(features_dim=128)
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)

    Args:
        features_dim: Output dimension of TCN extractor
        num_layers: Number of temporal blocks in TCN
        kernel_size: Kernel size for convolutions
        dropout: Dropout rate
        net_arch: Network architecture for policy/value heads (default: [64, 64])

    Returns:
        Dict suitable for PPO policy_kwargs parameter
    """
    if net_arch is None:
        net_arch = [dict(pi=[64, 64], vf=[64, 64])]

    return dict(
        features_extractor_class=TCNExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        ),
        net_arch=net_arch,
    )


# Fallback for when torch is not available
if not TORCH_AVAILABLE:
    def create_tcn_policy_kwargs(*args, **kwargs) -> dict:
        logger.warning("PyTorch not available; TCN policy kwargs will use default MLP")
        return {}
