# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch ViT model."""


import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

class PatchLogisticRegression(nn.Module):
    def __init__(self, config, image_height):
        super().__init__()
        self.config = config


        image_size = (image_height, config.image_width)
        patch_size = (int(config.patch_size[0] * image_height), config.patch_size[1])
        # patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        # self.activation = nn.GELU()
        # self.layer_norm = nn.GroupNorm(num_groups=hidden_size, num_channels=hidden_size, affine=True)

        # Classifier: single linear layer
        self.classifier = nn.Linear(800, config.num_labels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )

        # embeddings = self.projection(pixel_values)
        # embeddings = self.layer_norm(embeddings)
        # embeddings = self.activation(embeddings)
        # embeddings = embeddings.flatten(2).transpose(1, 2)
        # embeddings = embeddings.reshape(batch_size, -1)
        embeddings = pixel_values.reshape(batch_size, -1)
        logits = self.classifier(embeddings)
        return logits
    


class MultiEncoder(nn.Module):
    """Multi-channel encoder that processes inputs through CSA and CCA layers."""
    
    def __init__(self, config) -> None:
        super().__init__()
        self.region_dict = config.region_dict
        self.region_list = list(self.region_dict.values())
        image_height = config.image_height
        self.image_height = image_height if isinstance(image_height, collections.abc.Iterable) else [image_height] * config.input_channels
        
        num_csa_layers = config.input_channels
        self.csa_layers = nn.ModuleList([
            PatchLogisticRegression(config, image_height=self.image_height[i])
            for i in range(num_csa_layers)
        ])

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """Process multi-channel input through the model."""
        # Input validation
        if input_values is None:
            raise ValueError("input_values cannot be None")
            
        if input_values.dim() != 4:
            raise ValueError(f"Expected input_values to have 4 dimensions (batch_size, channels, height, width), got {input_values.dim()}")
            
        batch_size, num_channels, height, width = input_values.shape
        expected_channels = sum(self.region_list)
        if height != expected_channels:
            raise ValueError(f"Expected input height to be {expected_channels}, got {height}")

        # Process CSA outputs - group and concatenate by region, using multiple CSA layers if needed
        csa_outputs_by_region = []
        begin = 0
        csa_layer_idx = 0
        region_outputs = []
        for region, region_channels in self.region_dict.items():
            num_csa_layers = region_channels // 8
            for _ in range(num_csa_layers):
                end = begin + 8
                csa_output = self.csa_layers[csa_layer_idx](input_values[:, :, begin:end, :])
                cls_token = csa_output
                region_outputs.append(cls_token)
                begin = end
                csa_layer_idx += 1

        output = torch.stack(region_outputs, dim=0)
        output = output.mean(dim=0)
        
        return output, []
    


class PureLogisticRegression(nn.Module):
    """
    Pure logistic regression baseline: flattens input and applies a linear layer.
    No CNN, normalization, or activation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.region_dict = config.region_dict
        self.region_list = list(self.region_dict.values())
        self.image_height = sum(self.region_list)
        self.image_width = config.image_width
        self.num_channels = config.num_channels
        self.num_labels = config.num_labels
        input_dim = self.num_channels * self.image_height * self.image_width
        self.classifier = nn.Linear(input_dim, self.num_labels)

        # image_size = (self.image_height, config.image_width)
        # patch_size = (int(config.patch_size[0] * 8), config.patch_size[1])
        # # patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # hidden_size = 384 # config.hidden_size
        # self.image_size = image_size
        # self.patch_size = patch_size

        # self.num_patches = num_patches
        # self.projection = nn.Conv2d(self.num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        # self.activation = nn.GELU()
        # self.layer_norm = nn.GroupNorm(num_groups=hidden_size, num_channels=hidden_size, affine=True)
        # self.classifier = nn.Linear(num_patches * hidden_size, self.num_labels)

    def forward(self, pixel_values: torch.Tensor, output_attentions: Optional[bool] = None) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        # Flatten all but batch dimension
        # embeddings = self.projection(pixel_values)
        # embeddings = self.layer_norm(embeddings)
        # embeddings = self.activation(embeddings)
        embeddings = pixel_values
        embeddings = embeddings.view(batch_size, -1)
        logits = self.classifier(embeddings)
        return logits, []
    
    