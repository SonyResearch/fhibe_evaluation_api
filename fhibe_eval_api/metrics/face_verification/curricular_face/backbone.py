# SPDX-License-Identifier: Apache-2.0
"""Module containing the curricular face backbone network."""

from collections import namedtuple
from typing import List, Tuple, Union

import torch
from torch import nn as nn

Block = namedtuple("Block", ["in_channel", "depth", "stride"])


class BottleneckIR(nn.Module):
    """Bottleneck IR.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride to be used for the convolutional layers in the residual
            branch.

    Attributes:
        shortcut_layer (nn.Module): A shortcut layer that can either perform max
            pooling or convolution.
        res_layer (nn.Module): The residual layer containing two convolutional layers.

    Methods:
        forward(input_tensor): Performs the forward pass of the block.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):  # noqa: D107
        super(BottleneckIR, self).__init__()
        self.shortcut_layer: Union[nn.MaxPool2d, nn.Sequential]
        if in_channels == out_channels:
            self.shortcut_layer = nn.MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=False,
            ),
            nn.PReLU(num_parameters=out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Bottleneck IR.

        Args:
            input_tensor (torch.Tensor): A torch.Tensor.

        Returns:
            A torch.Tensor.
        """
        shortcut = self.shortcut_layer(input_tensor)
        res = self.res_layer(input_tensor)

        return res + shortcut


def resnet_block(
    in_channels: int, out_channels: int, num_blocks: int, stride: int = 2
) -> List[Block]:
    """Make ResNet blocks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_blocks (int): Number of blocks to create.
        stride (int): The strid of the first block.

    Returns:
        A list of ResNet Block named tuples.
    """
    return [Block(in_channels, out_channels, stride)] + [
        Block(out_channels, out_channels, 1) for i in range(num_blocks - 1)
    ]


def resnet_blocks() -> List[List[Block]]:
    """Make list of lists of ResNet blocks.

    Returns:
        A list of lists of ResNet Block named tuples.
    """
    return [
        resnet_block(in_channels=64, out_channels=64, num_blocks=3),
        resnet_block(in_channels=64, out_channels=128, num_blocks=13),
        resnet_block(in_channels=128, out_channels=256, num_blocks=30),
        resnet_block(in_channels=256, out_channels=512, num_blocks=3),
    ]


class IR101(nn.Module):
    def __init__(self) -> None:
        """Initializes an IR-101 model that expects inputs  sized [112, 112]."""
        super(IR101, self).__init__()

        blocks = resnet_blocks()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False),
        )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    BottleneckIR(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = nn.Sequential(*modules)

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass of the IR-101 model.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size,
                channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output_tensor (torch.Tensor): The output tensor of shape
                    (batch_size, embedding_size).
                - conv_out (torch.Tensor): The output of the last convolutional layer
                    before the output layer of shape (batch_size, num_channels, height,
                    width).
        """
        input_features = self.input_layer(input_tensor)
        body_features = self.body(input_features)
        conv_out = body_features.view(body_features.shape[0], -1)
        output_tensor = self.output_layer(body_features)

        return output_tensor, conv_out
