"""CrackMamba encoder, feature fusion, and segmentation heads."""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from .Module.CrackMamba_G import SS2D_continous
from .Module.CrackMamba_L import SS2D_local


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class GELU(nn.Module):
    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))


class LayerNorm(nn.Module):
    """Normalize the channel dimension of a four-dimensional feature map.

    The default channels_first layout is [B, C, H, W].
    The channels_last layout is [B, H, W, C].
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class VSSBlock(nn.Module):
    """Combine global selective scan features with a residual projection."""
    def __init__(
        self,
        hidden_dim: int = 0,
        output_dim: int = 0,
        drop_path: float = 0,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.self_attention = SS2D_continous(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)

        self.conv1 = nn.Conv2d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, input: torch.Tensor, weight):

        x_global = self.self_attention(input, weight0 = weight)
        output = self.conv1(input + x_global)

        return  output


class VSSBlock_local(nn.Module):
    """Combine local selective scan features with a residual projection."""
    def __init__(
        self,
        hidden_dim: int = 0,
        output_dim: int = 0,
        drop_path: float = 0,
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.self_attention_local = SS2D_local(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)

        self.conv1 = nn.Conv2d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, input: torch.Tensor):

        x_local = self.self_attention_local(input)
        output = self.conv1(input + x_local)

        return  output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out2 = torch.mean(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out, avg_out2], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)  # Reduce the channel dimension.
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)  # Restore the channel dimension.
        self.sigmoid = nn.Sigmoid()
        self.GELU = GELU()

    def forward(self, x):
        avg_out = self.fc2(self.GELU(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.GELU(self.fc1(self.max_pool(x))))
        avg_out2 = self.fc2(self.GELU(self.fc1(self.avg_pool(x))))

        out = avg_out + max_out + avg_out2
        return self.sigmoid(out)


class Stage1(nn.Module):
    """Project input channels and downsample the spatial resolution by two."""
    def __init__(self, in_ch, out_ch, Active):
        super(Stage1, self).__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'Hswish':
            Active_Layer = Hswish

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride=1, padding=1)
        self.norm1 = LayerNorm(out_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            Active_Layer(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, Active):
        super(DoubleConv, self).__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'Hswish':
            Active_Layer = Hswish

        self.conv1= nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = LayerNorm(in_ch)
        self.conv2= nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            Active_Layer(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


class Upsample_block(nn.Module):
    def __init__(self, in_ch, out_ch, s_factor):
        super(Upsample_block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = LayerNorm(out_ch)
        self.act = Hswish(inplace=True)
        self.upsample = nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.upsample(x)
        return x


class Fusion_module(nn.Module):
    def __init__(self, in_ch, out_ch, Active):
        super(Fusion_module, self).__init__()

        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'Hswish':
            Active_Layer = Hswish

        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1 = LayerNorm(in_ch)
        self.act1 = Active_Layer(inplace=True)

        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False)
        self.norm2 = LayerNorm(in_ch)

        self.conv3 = nn.Conv2d(in_ch, in_ch//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = LayerNorm(in_ch//2)
        self.act3 = Active_Layer(inplace=True)

        self.conv4 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False)
        self.norm4 = LayerNorm(in_ch)

        self.conv5 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm5 = LayerNorm(out_ch)
        self.act5 = Active_Layer(inplace=True)

        self.ca = ChannelAttention(in_ch//2)
        self.sa = SpatialAttention()

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        res = x
        x = self.conv2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)

        out1 = self.ca(x)
        out2 = self.sa(x)
        out1 = out1 * x
        out2 = out2 * x
        out = torch.cat([out1, out2], dim=1)

        out = out + res
        out = self.conv4(out)
        out = self.norm4(out)

        out = self.conv5(out)
        out = self.norm5(out)
        out = self.act5(out)

        return out


class Side_output_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Side_output_block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = LayerNorm(64)
        self.conv2 = nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


class Output_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Output_block, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def make_weight(B, C, H, W):
        """Create flattened serpentine scan indices for the given spatial dimensions."""
        weight = torch.arange(H*W).reshape(1,1,H,W)

        for i in range(H):
            if i % 2 == 1:
                weight[0,0,i,:] = torch.flip(weight[0,0,i,:], dims=[0])
        weight = weight.reshape(1, 1, 1, -1)
        return weight


def make_33_index():
    """Return the legacy 3 x 3 scan permutation and its inverse."""
    index_33_change = torch.tensor([7,6,3,0,1,2,5,8,4]).reshape(1, 1, 1, -1)
    index_33_restore = torch.tensor([3, 4, 5, 2, 8, 6, 1, 0, 7]).reshape(1, 1, 1, -1)

    return index_33_change, index_33_restore


class CrackSeU(nn.Module):
    """CrackMamba segmentation model with an auxiliary supervision head."""
    def __init__(self, in_ch, out_ch):
        super(CrackSeU, self).__init__()

        self.filter = [32, 64, 128, 256, 512]

        # Legacy scan indices retained in the model state.
        scale = 1
        for _ in range(5):
            scale = scale * 2
            weight = make_weight(B=1, C=1, H=512//scale, W=512//scale)
            self.register_buffer("weight_"+str(scale), weight)

        index_33_change, index_33_restore = make_33_index()
        self.register_buffer("index_33_change", index_33_change)
        self.register_buffer("index_33_restore", index_33_restore)

        # Encoder stages.
        self.conv1 = Stage1(in_ch, self.filter[0],'GELU')  # Half the input resolution.
        self.pool1 = nn.MaxPool2d(2)  # One quarter of the input resolution.
        self.mamba2 = VSSBlock(hidden_dim=self.filter[0], output_dim=self.filter[1])
        self.pool2 = nn.MaxPool2d(2)  # One eighth of the input resolution.
        self.mamba3 = VSSBlock(hidden_dim=self.filter[1], output_dim=self.filter[2])
        self.pool3 = nn.MaxPool2d(2)  # One sixteenth of the input resolution.
        self.mamba4 = VSSBlock_local(hidden_dim=self.filter[2], output_dim=self.filter[3])

        self.pool4 = nn.MaxPool2d(2)  # One thirty-second of the input resolution.

        self.mamba5 = VSSBlock_local(hidden_dim=self.filter[3], output_dim=self.filter[4])

        # Upsampling paths.
        self.upsample_5_3 = Upsample_block(512, 256, 4)
        self.upsample_4_3 = Upsample_block(256, 128, 2)
        self.upsample_3_1 = Upsample_block(128*2, 128, 4)
        self.upsample_2_1 = Upsample_block(64, 32, 2)

        # Feature fusion modules.
        self.fusion1 = Fusion_module(128*4, 128*2,'Hswish')
        self.fusion2 = Fusion_module(64, 64,'Hswish')
        self.fusion3 = Fusion_module(192, 32,'Hswish')

        # Auxiliary segmentation head.
        self.min_2x = Side_output_block(128, out_ch)

        # Main segmentation head.
        self.upsample_1_0 = Output_block(32, out_ch)


    def forward(self, x):
        """Return main and auxiliary segmentation probabilities.

        For spatial sizes divisible by 32, the main output matches the input
        resolution and the auxiliary output has half the input resolution.
        """
        B, C, H, W = x.shape
        weight_4 = self.weight_4.expand(B, -1, self.filter[0], -1)  # Legacy scan index layout: [B, 1, C, L].
        weight_8 = self.weight_8.expand(B, -1, self.filter[1], -1)

        # Extract features at progressively lower resolutions.
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.mamba2(p1, weight_4)
        p2 = self.pool2(c2)
        c3 = self.mamba3(p2, weight_8)
        p3 = self.pool3(c3)
        c4 = self.mamba4(p3)
        p4 = self.pool4(c4)
        c5 = self.mamba5(p4)

        # Fuse encoder features across resolutions.
        up_5_3 = self.upsample_5_3(c5)
        up_4_3 = self.upsample_4_3(c4)
        merge1 = torch.cat([up_5_3, up_4_3, c3], dim=1)
        fusion1_out = self.fusion1(merge1)

        up_2_1 = self.upsample_2_1(c2)
        merge2 = torch.cat([up_2_1, c1], dim=1)
        fusion2_out = self.fusion2(merge2)

        up_3_1 = self.upsample_3_1(fusion1_out)
        merge3 = torch.cat([up_3_1, fusion2_out], dim=1)
        fusion3_out = self.fusion3(merge3)

        # Produce the main segmentation output.
        up_1_0 = self.upsample_1_0(fusion3_out)

        # Produce the auxiliary output at half the input resolution.
        min_2x_out = self.min_2x(up_3_1)
        min_2x_out = nn.Sigmoid()(min_2x_out)

        # Convert logits to segmentation probabilities.
        out = nn.Sigmoid()(up_1_0)

        return out, min_2x_out


class CrackSeU_Inference(nn.Module):
    """CrackMamba inference architecture with the main segmentation head."""
    def __init__(self, in_ch, out_ch):
        super(CrackSeU_Inference, self).__init__()

        self.filter = [32, 64, 128, 256, 512]

        # Legacy scan indices retained in the model state.
        scale = 1
        for _ in range(5):
            scale = scale * 2
            weight = make_weight(B=1, C=1, H=512//scale, W=512//scale)
            self.register_buffer("weight_"+str(scale), weight)

        index_33_change, index_33_restore = make_33_index()
        self.register_buffer("index_33_change", index_33_change)
        self.register_buffer("index_33_restore", index_33_restore)

        # Encoder stages.
        self.conv1 = Stage1(in_ch, self.filter[0],'GELU')  # Half the input resolution.
        self.pool1 = nn.MaxPool2d(2)  # One quarter of the input resolution.
        self.mamba2 = VSSBlock(hidden_dim=self.filter[0], output_dim=self.filter[1])
        self.pool2 = nn.MaxPool2d(2)  # One eighth of the input resolution.
        self.mamba3 = VSSBlock(hidden_dim=self.filter[1], output_dim=self.filter[2])
        self.pool3 = nn.MaxPool2d(2)  # One sixteenth of the input resolution.
        self.mamba4 = VSSBlock_local(hidden_dim=self.filter[2], output_dim=self.filter[3])

        self.pool4 = nn.MaxPool2d(2)  # One thirty-second of the input resolution.

        self.mamba5 = VSSBlock_local(hidden_dim=self.filter[3], output_dim=self.filter[4])

        # Upsampling paths.
        self.upsample_5_3 = Upsample_block(512, 256, 4)
        self.upsample_4_3 = Upsample_block(256, 128, 2)
        self.upsample_3_1 = Upsample_block(128*2, 128, 4)
        self.upsample_2_1 = Upsample_block(64, 32, 2)

        # Feature fusion modules.
        self.fusion1 = Fusion_module(128*4, 128*2,'Hswish')
        self.fusion2 = Fusion_module(64, 64,'Hswish')
        self.fusion3 = Fusion_module(192, 32,'Hswish')

        # Main segmentation head.
        self.upsample_1_0 = Output_block(32, out_ch)


    def forward(self, x):

        """Return segmentation probabilities for the input image batch."""
        B, C, H, W = x.shape
        weight_4 = self.weight_4.expand(B, -1, self.filter[0], -1)  # Legacy scan index layout: [B, 1, C, L].
        weight_8 = self.weight_8.expand(B, -1, self.filter[1], -1)

        # Extract features at progressively lower resolutions.
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.mamba2(p1, weight_4)
        p2 = self.pool2(c2)
        c3 = self.mamba3(p2, weight_8)
        p3 = self.pool3(c3)
        c4 = self.mamba4(p3)
        p4 = self.pool4(c4)
        c5 = self.mamba5(p4)

        # Fuse encoder features across resolutions.
        up_5_3 = self.upsample_5_3(c5)
        up_4_3 = self.upsample_4_3(c4)
        merge1 = torch.cat([up_5_3, up_4_3, c3], dim=1)
        fusion1_out = self.fusion1(merge1)

        up_2_1 = self.upsample_2_1(c2)
        merge2 = torch.cat([up_2_1, c1], dim=1)
        fusion2_out = self.fusion2(merge2)

        up_3_1 = self.upsample_3_1(fusion1_out)
        merge3 = torch.cat([up_3_1, fusion2_out], dim=1)
        fusion3_out = self.fusion3(merge3)

        # Produce the main segmentation output.
        up_1_0 = self.upsample_1_0(fusion3_out)

        # Convert logits to segmentation probabilities.
        out = nn.Sigmoid()(up_1_0)
        return out
