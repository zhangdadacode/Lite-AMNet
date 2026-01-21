import math

import numpy as np
import torch
import torch.nn as nn

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1, multi_scales=None):
	"""
	Parameter description:
	    in_channels: Number of input feature channels
	    out_channels: Number of output feature channels
	    indices: Neighborhood index tensor with shape [num_nodes, seq_length]
	    dim: Dimension along which the operation is applied, default is 1
	    multi_scales: List of multi-scale neighborhood indices; if None, a single scale is used
	"""

        super(DSConv, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        if multi_scales is not None:

            self.indices_list = multi_scales
            self.num_scales = len(multi_scales)

            seq_lengths = [idx.size(1) for idx in multi_scales]
            assert len(set(seq_lengths)) == 1, 
            self.seq_length = seq_lengths[0]
            self.is_multi_scale = True
        else:

            self.indices = indices
            self.seq_length = indices.size(1)
            self.is_multi_scale = False
            self.num_scales = 1

        self.kernel_size = int(math.sqrt(self.seq_length))

        if self.is_multi_scale:
     
            self._init_multi_scale_layers()
        else:
           
            self._init_single_scale_layers()

        self.reset_parameters()

    def _init_single_scale_layers(self):

        self.spatial_layer = nn.Conv2d(
            self.in_channels, self.in_channels, self.kernel_size,
            stride=1, padding=0, groups=self.in_channels, bias=False
        )

        self.channel_layer = nn.Linear(self.in_channels, self.out_channels, bias=False)

    def _init_multi_scale_layers(self):

        self.spatial_layer = nn.Conv2d(
            self.in_channels, self.in_channels, self.kernel_size,
            stride=1, padding=0, groups=self.in_channels, bias=False
        )

    
        self.channel_attention = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 4),  
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels // 4, self.in_channels),  
            nn.Sigmoid() 
        )


        self.fusion_conv = nn.Conv1d(
            self.in_channels * self.num_scales, self.out_channels, 1, bias=False
        )

        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)

    def reset_parameters(self):
        if hasattr(self, 'spatial_layer'):
            nn.init.kaiming_uniform_(self.spatial_layer.weight, nonlinearity='relu')

        if hasattr(self, 'channel_layer'):
            nn.init.xavier_uniform_(self.channel_layer.weight)

        if hasattr(self, 'fusion_conv'):
            nn.init.kaiming_uniform_(self.fusion_conv.weight, nonlinearity='relu')

        if hasattr(self, 'channel_attention'):
            for layer in self.channel_attention:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def process_single_scale(self, x, indices):
	"""
	Process feature extraction at a single scale.

	Args:
	    x: Input tensor with shape [batch_size, num_nodes, in_channels]
	    indices: Neighborhood indices with shape [num_nodes, seq_length]

	Returns:
	    Processed features with shape [batch_size, num_nodes, in_channels]
	"""

        bs = x.size(0)
        n_nodes = indices.size(0)


        x_selected = torch.index_select(x, self.dim, indices.to(x.device).view(-1))


        x_reshaped = x_selected.view(bs * n_nodes, self.seq_length, -1).transpose(1, 2)
        x_spatial = x_reshaped.view(bs * n_nodes, self.in_channels,
                                    self.kernel_size, self.kernel_size)

        spatial_out = self.spatial_layer(x_spatial)
        spatial_out = spatial_out.view(bs, n_nodes, -1)

        return spatial_out

    def forward(self, x):
	"""
	Forward pass.

	Args:
	    x: Input tensor with shape [batch_size, num_nodes, in_channels]

	Returns:
	    output: Output tensor with shape [batch_size, num_nodes, out_channels]
	"""

        if self.is_multi_scale:
            return self._forward_multi_scale(x)
        else:
            return self._forward_single_scale(x)

    def _forward_single_scale(self, x):
    
        bs = x.size(0)
        n_nodes = self.indices.size(0)

        x = torch.index_select(x, self.dim, self.indices.to(x.device).view(-1))

        x = x.view(bs * n_nodes, self.seq_length, -1).transpose(1, 2)
        x = x.view(x.size(0), x.size(1), self.kernel_size, self.kernel_size)
        x = self.spatial_layer(x).view(bs, n_nodes, -1)

        x = self.channel_layer(x)
        return x

    def _forward_multi_scale(self, x):

        bs = x.size(0)
        multi_scale_features = []


        for scale_idx, indices in enumerate(self.indices_list):

            scale_feat = self.process_single_scale(x, indices)

            channel_attn = self.channel_attention(scale_feat.mean(dim=1))  # [bs, in_channels]
            
            calibrated_feat = scale_feat * channel_attn.unsqueeze(1)  # [bs, n_nodes, in_channels]

            weighted_feat = calibrated_feat * self.scale_weights[scale_idx]
            multi_scale_features.append(weighted_feat)

        fused_features = torch.cat(multi_scale_features, dim=-1)  # [bs, n_nodes, in_channels * num_scales]

        output = self.fusion_conv(fused_features.transpose(1, 2)).transpose(1, 2)  # [bs, n_nodes, out_channels]

        return output

    def __repr__(self):
        if self.is_multi_scale:
            return '{}({}, {}, scales={}, seq_length={})'.format(
                self.__class__.__name__,
                self.in_channels,
                self.out_channels,
                self.num_scales,
                self.seq_length
            )
        else:
            return '{}({}, {}, seq_length={})'.format(
                self.__class__.__name__,
                self.in_channels,
                self.out_channels,
                self.seq_length
            )
