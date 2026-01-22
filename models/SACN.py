import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format


def channel_shuffle_manual(x, groups):
    batch_size, num_channels, sequence_length = x.size()
    if num_channels % groups != 0:
        raise ValueError('Input channels must be divisible by the number of groups.')
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, sequence_length)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, sequence_length)
    return x

class LocalBranch(nn.Module):
    def __init__(self, in_channels, groups=4): 
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        self.groups = groups
        self.conv1x1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):

        residual = x #(10,256,21)

        x = self.conv1x1(x) #(10,256,21)
        x = channel_shuffle_manual(x, self.groups) #(10,256,21)
        x = self.conv3x3(x) #(10,256,21)

        # 应用 LayerNorm
        x = x.permute(0, 2, 1)  # [B, C, N] -> [B, N, C]
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # [B, N, C] -> [B, C, N]

        return x + residual 


# ------------------- GlobalBranch -------------------
class GlobalBranch(nn.Module):
    def __init__(self, in_channels, dilation_rate=1):  # dilation_rate:2 、优化为1
        super().__init__()

        self.depthwise_dconv_q = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding='same',
                                           dilation=dilation_rate, groups=in_channels)
        self.pointwise_dconv_q = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        self.depthwise_dconv_k = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding='same',
                                           dilation=dilation_rate, groups=in_channels)
        self.pointwise_dconv_k = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        self.depthwise_conv1x1_v = nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        self.pointwise_conv1x1_v = nn.Conv1d(in_channels, in_channels, kernel_size=1)


        self.norm = nn.LayerNorm(in_channels)
        self.return_attention = False
        self.last_attention = None
    def forward(self, x):
    
        v_residual = self.depthwise_conv1x1_v(x)
        v_residual = self.pointwise_conv1x1_v(v_residual)

        q = self.depthwise_dconv_q(x)
        q = self.pointwise_dconv_q(q)

        k = self.depthwise_dconv_k(x)
        k = self.pointwise_dconv_k(k)

        v = self.depthwise_dconv_v(v_residual)
        v = self.pointwise_dconv_v(v)

        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        d_k = k.size(-1)
        attention_map = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attention_map = F.softmax(attention_map, dim=-1)

        try:
            self.last_attention = attention_map.detach().cpu()
        except Exception:
            self.last_attention = attention_map.cpu()

        attention_output = attention_output.permute(0, 2, 1)
        attention_output = self.norm(attention_output)
        attention_output = attention_output.permute(0, 2, 1)

        output = attention_output + v_residual

        if self.return_attention:
            return output, attention_map
        else:
            return output


class SACN(nn.Module):
    def __init__(self, in_channels, groups=4, dilation_rate=1):
        super().__init__()
        self.local_branch = LocalBranch(in_channels, groups)
        self.global_branch = GlobalBranch(in_channels, dilation_rate)
        self.apply(self._initialize_weights)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        local_out = self.local_branch(x_permuted)
        global_out = self.global_branch(x_permuted)
        combined_out = local_out + global_out
        output = combined_out.permute(0, 2, 1)
        return output

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
