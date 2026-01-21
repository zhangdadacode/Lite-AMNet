import math
import torch.nn as nn
import torch
from models.SCAF import SCAF
from models.layer import make_linear_layers
from conv import DSConv
from models.AGCN import GCNfction
from models.SACN import SACN
import torch.nn.functional as F


# Basic modules
class Reorg(nn.Module):
    dump_patches = True

    def __init__(self):
        """Reorg layer to re-organize spatial dim and channel dim
        """
        super(Reorg, self).__init__()

    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1], ss[2] // 2, 2, ss[3] // 2, 2). \
            permute(0, 1, 3, 5, 2, 4).contiguous().view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out


def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, relu=True, group=1):
    """Conv block

    Args:
        channel_in (int): input channel size
        channel_out (int): output channel size
        ks (int, optional): kernel size. Defaults to 1.
        stride (int, optional): Defaults to 1.
        padding (int, optional): Defaults to 0.
        dilation (int, optional): Defaults to 1.
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.
        group (int, optional): group conv parameter. Defaults to 1.

    Returns:
        Sequential: a block with bn and relu
    """
    _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation,
                      bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        sequence.append(nn.ReLU())

    return nn.Sequential(*sequence)


def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True):
    """Fully connected block

    Args:
        channel_in (int): input channel size
        channel_out (_type_): output channel size
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.

    Returns:
        Sequential: a block with bn and relu
    """
    _linear = nn.Linear
    sequence = [_linear(channel_in, channel_out, bias=bias)]

    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        sequence.append(nn.Hardtanh(0, 4))

    return nn.Sequential(*sequence)


class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, has_half_out=False, num3x3=1):
        """Init a depth-wise sparable convolution

        Args:
            channel_in (int): input channel size
            channel_out (_type_): output channel size
            stride (int, optional): conv stride. Defaults to 1.
            has_half_out (bool, optional): whether output intermediate result. Defaults to False.
            num3x3 (int, optional): amount of 3x3 conv layer. Defaults to 1.
        """
        super(mobile_unit, self).__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=1, padding=1, group=channel_in),
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.has_half_out = has_half_out

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = out + x
        if self.has_half_out:
            return half_out, out
        else:
            return out


def Pool(x, trans, dim=1):
    """Upsample a mesh

    Args:
        x (tensor): input tensor, BxNxD
        trans (tuple): upsample indices and valus
        dim (int, optional): upsample axis. Defaults to 1.

    Returns:
        tensor: upsampled tensor, BxN'xD
    """
    row, col, value = trans[0].to(x.device), trans[1].to(x.device), trans[2].to(x.device)
    value = value.unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out2 = torch.zeros(x.size(0), row.size(0) // 3, x.size(-1)).to(x.device)
    idx = row.unsqueeze(0).unsqueeze(-1).expand_as(out)
    out2 = torch.scatter_add(out2, dim, idx, out)
    return out2


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, meshconv=DSConv):
        """Init a spiral conv block

        Args:
            in_channels (int): input feature dim
            out_channels (int): output feature dim
            indices (tensor): neighbourhood of each hand vertex
            meshconv (optional): conv method. Defaults to DSConv.
        """
        super(SpiralDeblock, self).__init__()
        self.conv = meshconv(in_channels, out_channels, indices)
        self.relu = nn.ReLU(inplace=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = self.relu(self.conv(out))
        return out


# Advanced modules
class Reg2DDecode3D(nn.Module):
    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=DSConv):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark
            meshconv (optional): conv method. Defaults to DSConv.
        """
        super(Reg2DDecode3D, self).__init__()

        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform

        self.num_vert = [u[0].size(0) // 3 for u in self.up_transform] + [self.up_transform[-1][0].size(0) // 6]

        self.uv_channel = uv_channel
        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1], 1, bn=False, relu=False)
        self.de_layer = nn.ModuleList()

        self.askc_fuse_feat = SCAF(channels=256, r=4)
        self.conv1x1_pre = conv_layer(128, 256, ks=1, stride=1, padding=0)  # pre_out_feat:128→256
        self.conv1x1_s1 = conv_layer(24, 256, ks=1, stride=1, padding=0)  # stack1_out_feat:24→256
        self.conv1x1_s2 = conv_layer(24, 256, ks=1, stride=1, padding=0)  # stack2_out_feat:24→256
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1],
                                                   self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(
                    SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1],
                                  meshconv=meshconv))

        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel]) * 0.01, requires_grad=True)
        self.verts = [49, 98, 195, 389] 
        self.dim = [256, 256, 128, 64] 
        self.GCNlist = nn.ModuleList()
        
        for i in range(4):
            self.GCNlist.append(GCNfction(self.verts[i], self.dim[i])) 
        self.CombinedNetwork = SACN(256, 4, dilation_rate=2)
        self.feat_fusion = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),  
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1) 
        )
        self.conv1x1 = nn.Conv1d(in_channels=688, out_channels=256, kernel_size=1)
        self.anchor_as_pre = AnchorAS(feat_dims=128, num_anchors=4)  # pre_out (128 channels)
        self.anchor_as_s1 = AnchorAS(feat_dims=24, num_anchors=4)  # stack1_out (24 channels)
        self.anchor_as_s2 = AnchorAS(feat_dims=24, num_anchors=4)  # stack2_out (24 channels)
        self.anchor_as_latent = AnchorAS(feat_dims=256, num_anchors=4)  # x (256 channels)

    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x, pre_out, stack1_out, stack2_out):
            """
            :param uv: （B,21,2）
            :param x:（B,256,4,4）
            :param pre_out:（B,128,64,64）
            :param stack1_out:（B,24,32,32）
            :param stack2_out:（B,24,16,16）
            :return: pred_final（B,778,3）
            """
            uv = torch.clamp((uv - 0.5) * 2, -1, 1) 
            x = self.de_layer_conv(x)

            ## Step 1: Multi-scale feature anchor sampling

            #pre sampling
            pre_out_feat = self.anchor_as_pre(pre_out, uv) 
            pre_out_feat_2d = self.conv1x1_pre(pre_out_feat.permute(0, 2, 1).unsqueeze(-1))
            #stack1_out sampling
            stack1_out_feat = self.anchor_as_s1(stack1_out, uv)  
            stack1_out_feat_2d = self.conv1x1_s1(stack1_out_feat.permute(0, 2, 1).unsqueeze(-1))
            #stack2_out sampling
            stack2_out_feat = self.anchor_as_s2(stack2_out, uv) 
            stack2_out_feat_2d = self.conv1x1_s2(stack2_out_feat.permute(0, 2, 1).unsqueeze(-1))
            # Use feature maps output by anchor attention sampling.
            latent_out = self.anchor_as_latent(x, uv)  
            latent_out_2d = latent_out.permute(0, 2, 1).unsqueeze(-1)

            # Step 2: Attention enhancement and multi-scale fusion
            attention_out = self.CombinedNetwork(latent_out) 
            attention_out_2d = attention_out.permute(0, 2, 1).unsqueeze(-1)
            fused_feat = latent_out_2d
            fused_feat = self.askc_fuse_feat(fused_feat, attention_out_2d)
            fused_feat = self.askc_fuse_feat(fused_feat, stack2_out_feat_2d)
            fused_feat = self.askc_fuse_feat(fused_feat, stack1_out_feat_2d)
            fused_feat = self.askc_fuse_feat(fused_feat, pre_out_feat_2d)
            fused_feat_1d = fused_feat.squeeze(-1).permute(0, 2, 1)  

            # Step 3: Feature-level fusion
            combined_feat = torch.cat([attention_out, fused_feat_1d], dim=2)  # [B,21,512]
            combined_feat = combined_feat.permute(0, 2, 1)  # [B,512,21]
            unified_feat = self.feat_fusion(combined_feat)  # [B,256,21]
            unified_feat = unified_feat.permute(0, 2, 1)  # [B,21,256]

           # Step 4: Single decoding path
            z = torch.bmm(self.upsample.repeat(unified_feat.size(0), 1, 1).to(unified_feat.device), unified_feat)
            num_features = len(self.de_layer)

            for i, layer in enumerate(self.de_layer):
                z_GCN = self.GCNlist[i](z)
                z = layer(z_GCN, self.up_transform[num_features - i - 1])
                
            pred_final = self.head(z)  # [B,778,3]

            return pred_final


class AnchorGenerator(nn.Module):

    def __init__(self, num_anchors=4, anchor_range=0.03):

        super().__init__()
        self.num_anchors = num_anchors
        self.anchor_range = anchor_range


        if num_anchors == 1:
        
            self.base_offsets = nn.Parameter(torch.zeros(1, 2), requires_grad=False)
        else:
            k = int(math.sqrt(num_anchors))
            assert k * k >= num_anchors, 
            x = torch.linspace(-anchor_range, anchor_range, k)
            y = torch.linspace(-anchor_range, anchor_range, k)
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            offsets = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:num_anchors]
            self.base_offsets = nn.Parameter(offsets, requires_grad=False) 

    def forward(self, uv):
	"""
        Args:
            uv: Raw joint coordinates (B, N, 2), normalized to [-1, 1] (image coordinates).
        Returns:
            anchors: Anchor coordinates for each joint (B, N, K, 2), where K=num_anchors.
        """
        B, N, _ = uv.shape

        anchors = uv.unsqueeze(2) + self.base_offsets.unsqueeze(0).unsqueeze(0)
        
        return torch.clamp(anchors, -1.0, 1.0)


class AnchorAttention(nn.Module):
    """Anchor attention weight calculation: Dynamic weighting based on feature similarity between anchors and joints."""

    def __init__(self, feat_dim=256, num_anchors=4):
        super().__init__()
        self.similarity_encoder = make_linear_layers(
            [feat_dim, feat_dim // 2, 1], 
            relu_final=False
        )
        self.num_anchors = num_anchors

    def forward(self, anchor_feats, joint_feat):
	"""
        Args:
            anchor_feats: Anchor-sampled features (B, N, K, C), where K=num_anchors.
            joint_feat: Raw joint features (B, N, C).
        Returns:
            attn_weights: Anchor attention weights (B, N, K), normalized via Softmax.
        """
        B, N, K, C = anchor_feats.shape
        diff = anchor_feats - joint_feat.unsqueeze(2)  
        raw_weights = self.similarity_encoder(diff).squeeze(-1)  
        return F.softmax(raw_weights, dim=-1)


class AnchorAS(nn.Module):
	"""
	2D Anchor-Assisted Attention Sampling Module: Combines anchor features with attention weighting.
	"""

    def __init__(self, feat_dims, num_anchors=4, anchor_range=0.03):
   
    """
    Args:
        feat_dims: Input feature map channels (adapts to features of different scales).
        num_anchors: Number of anchors per joint.
        anchor_range: Anchor distribution range.
    """
        super().__init__()
        self.anchor_gen = AnchorGenerator(num_anchors, anchor_range)
        self.anchor_attn = AnchorAttention(feat_dims, num_anchors)
        self.num_anchors = num_anchors

    def forward(self, feat_map, uv):

        
        B, C, H, W = feat_map.shape
        N = uv.shape[1]
        K = self.num_anchors
        anchors = self.anchor_gen(uv)

        anchors_flat = anchors.view(B, N * K, 1, 2)
        anchor_feats = F.grid_sample(
            feat_map, anchors_flat, align_corners=True
        ).squeeze(-1)  

        anchor_feats = anchor_feats.view(B, C, N, K).permute(0, 2, 3, 1)  # (B, N, K, C)

        joint_feats = F.grid_sample(
            feat_map, uv.unsqueeze(2), align_corners=True
        ).squeeze(-1).permute(0, 2, 1)  # (B, N, C)

        attn_weights = self.anchor_attn(anchor_feats, joint_feats)  # (B, N, K)


        weighted_feat = torch.sum(
            anchor_feats * attn_weights.unsqueeze(-1),  # (B, N, K, C) * (B, N, K, 1)
            dim=2 
        ) 

        return weighted_feat

