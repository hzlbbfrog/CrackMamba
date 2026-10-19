"""Deepcrack G-only architecture with strict legacy checkpoint compatibility.

The auxiliary head remains in the state dictionary but is omitted from inference.
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .global_scan import SS2D_continous, SS2D_forward


def load_model(checkpoint, device="cpu", scan_backend="mamba"):
    """Load all trained parameters and buffers strictly, then select a scan backend."""
    if scan_backend not in {"mamba", "reference"}:
        raise ValueError("scan_backend must be 'mamba' or 'reference'")
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("The checkpoint must contain a nonempty tensor state dictionary")
    cleaned = {}
    for key, value in state.items():
        name = key[7:] if key.startswith("module.") else key
        if name in cleaned or not isinstance(value, torch.Tensor):
            raise ValueError(f"Invalid or duplicate checkpoint entry: {key}")
        cleaned[name] = value
    model = CrackSeU()
    model.load_state_dict(cleaned, strict=True)
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    for module in model.modules():
        if isinstance(module, SS2D_forward):
            module.scan_backend = scan_backend
    model.eval().requires_grad_(False)
    return model.to(device)

class Hswish(nn.Module):
    """Hard-swish activation used by the trained decoder."""

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

class GELU(nn.Module):
    """The trained model's tanh approximation of GELU."""

    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    """Layer normalization with channels-first and channels-last layouts."""

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_first'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class VSSBlock(nn.Module):
    """A global selective-scan block with residual feature projection."""

    def __init__(self, hidden_dim: int=0, output_dim: int=0, drop_path: float=0, attn_drop_rate: float=0, d_state: int=16, **kwargs):
        super().__init__()
        self.self_attention = SS2D_continous(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.conv1 = nn.Conv2d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, input: torch.Tensor, weight, flag):
        x_mamba = self.self_attention(input, weight0=weight, flag=flag)
        output1 = input + x_mamba
        output = self.conv1(output1)
        return output

class SpatialAttention(nn.Module):
    """Spatial feature modulation."""

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
    """Channel-wise feature modulation."""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.GELU = GELU()

    def forward(self, x):
        avg_out = self.fc2(self.GELU(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.GELU(self.fc1(self.max_pool(x))))
        avg_out2 = self.fc2(self.GELU(self.fc1(self.avg_pool(x))))
        out = avg_out + max_out + avg_out2
        return self.sigmoid(out)

class Stage1(nn.Module):
    """The stride-two convolutional stem."""

    def __init__(self, in_ch, out_ch, Active):
        super(Stage1, self).__init__()
        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'Hswish':
            Active_Layer = Hswish
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm1 = LayerNorm(out_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False), Active_Layer(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        return x

class DoubleConv(nn.Module):
    """Convolutional feature extraction used by the original architecture."""

    def __init__(self, in_ch, out_ch, Active):
        super(DoubleConv, self).__init__()
        if Active == 'ReLU':
            Active_Layer = nn.ReLU
        elif Active == 'GELU':
            Active_Layer = GELU
        elif Active == 'Hswish':
            Active_Layer = Hswish
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = LayerNorm(in_ch)
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False), Active_Layer(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x

class Upsample_block(nn.Module):
    """The checkpoint-compatible decoder upsampling block."""

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
    """Fuse multiscale features using channel and spatial attention."""

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
        self.conv3 = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = LayerNorm(in_ch // 2)
        self.act3 = Active_Layer(inplace=True)
        self.conv4 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False)
        self.norm4 = LayerNorm(in_ch)
        self.conv5 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm5 = LayerNorm(out_ch)
        self.act5 = Active_Layer(inplace=True)
        self.ca = ChannelAttention(in_ch // 2)
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
    """Retain auxiliary-head parameters for strict checkpoint loading."""

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
    """Upsample decoder features into full-resolution logits."""

    def __init__(self, in_ch, out_ch):
        super(Output_block, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

def make_weight(B, C, H, W):
    weight = torch.arange(H * W).reshape(1, 1, H, W)
    for i in range(H):
        if i % 2 == 1:
            weight[0, 0, i, :] = torch.flip(weight[0, 0, i, :], dims=[0])
    weight = weight.reshape(1, 1, 1, -1)
    return weight

class CrackSeU(nn.Module):
    """Deepcrack global-only network retaining the original checkpoint keys."""

    def __init__(self, in_ch=3, out_ch=1, image_h1=480, image_h2=384, image_h3=544, image_w3=384):
        super(CrackSeU, self).__init__()
        self.filter = [32, 64, 128, 256, 512]
        scale = 1
        self.image_h1 = image_h1
        self.image_h2 = image_h2
        self.image_h3 = image_h3
        self.image_w3 = image_w3
        for _ in range(5):
            scale = scale * 2
            weight_h1 = make_weight(B=1, C=1, H=self.image_h1 // scale, W=self.image_h1 // scale)
            weight_h2 = make_weight(B=1, C=1, H=self.image_h2 // scale, W=self.image_h2 // scale)
            weight_h3w3 = make_weight(B=1, C=1, H=self.image_h3 // scale, W=self.image_w3 // scale)
            weight_w3h3 = make_weight(B=1, C=1, H=self.image_w3 // scale, W=self.image_h3 // scale)
            self.register_buffer('weight_h1_' + str(scale), weight_h1)
            self.register_buffer('weight_h2_' + str(scale), weight_h2)
            self.register_buffer('weight_hw_' + str(scale), weight_h3w3)
            self.register_buffer('weight_wh_' + str(scale), weight_w3h3)
        self.conv1 = Stage1(in_ch, self.filter[0], 'GELU')
        self.pool1 = nn.MaxPool2d(2)
        self.mamba2 = VSSBlock(hidden_dim=self.filter[0], output_dim=self.filter[1])
        self.pool2 = nn.MaxPool2d(2)
        self.mamba3 = VSSBlock(hidden_dim=self.filter[1], output_dim=self.filter[2])
        self.pool3 = nn.MaxPool2d(2)
        self.mamba4 = VSSBlock(hidden_dim=self.filter[2], output_dim=self.filter[3])
        self.pool4 = nn.MaxPool2d(2)
        self.mamba5 = VSSBlock(hidden_dim=self.filter[3], output_dim=self.filter[4])
        self.upsample_5_3 = Upsample_block(512, 256, 4)
        self.upsample_4_3 = Upsample_block(256, 128, 2)
        self.upsample_3_1 = Upsample_block(128 * 2, 128, 4)
        self.upsample_2_1 = Upsample_block(64, 32, 2)
        self.fusion1 = Fusion_module(128 * 4, 128 * 2, 'Hswish')
        self.fusion2 = Fusion_module(64, 64, 'Hswish')
        self.fusion3 = Fusion_module(192, 32, 'Hswish')
        self.min_2x = Side_output_block(128, out_ch)
        self.upsample_1_0 = Output_block(32, out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        if not torch.onnx.is_in_onnx_export():
            supported = {(self.image_h1, self.image_h1), (self.image_h2, self.image_h2),
                         (self.image_h3, self.image_w3), (self.image_w3, self.image_h3)}
            if C != 3 or B < 1 or (H, W) not in supported:
                raise ValueError(f"Expected RGB NCHW input with spatial shape in {sorted(supported)}")
        flag = 0
        if H == self.image_h1 and W == self.image_h1:
            weight_4 = [self.weight_h1_4.expand(B, -1, self.filter[0], -1), self.weight_h1_4.expand(B, -1, self.filter[0], -1)]
            weight_8 = [self.weight_h1_8.expand(B, -1, self.filter[1], -1), self.weight_h1_8.expand(B, -1, self.filter[1], -1)]
            weight_16 = [self.weight_h1_16.expand(B, -1, self.filter[2], -1), self.weight_h1_16.expand(B, -1, self.filter[2], -1)]
            weight_32 = [self.weight_h1_32.expand(B, -1, self.filter[3], -1), self.weight_h1_32.expand(B, -1, self.filter[3], -1)]
        elif H == self.image_h2 and W == self.image_h2:
            weight_4 = [self.weight_h2_4.expand(B, -1, self.filter[0], -1), self.weight_h2_4.expand(B, -1, self.filter[0], -1)]
            weight_8 = [self.weight_h2_8.expand(B, -1, self.filter[1], -1), self.weight_h2_8.expand(B, -1, self.filter[1], -1)]
            weight_16 = [self.weight_h2_16.expand(B, -1, self.filter[2], -1), self.weight_h2_16.expand(B, -1, self.filter[2], -1)]
            weight_32 = [self.weight_h2_32.expand(B, -1, self.filter[3], -1), self.weight_h2_32.expand(B, -1, self.filter[3], -1)]
        else:
            weight_4 = [self.weight_hw_4.expand(B, -1, self.filter[0], -1), self.weight_wh_4.expand(B, -1, self.filter[0], -1)]
            weight_8 = [self.weight_hw_8.expand(B, -1, self.filter[1], -1), self.weight_wh_8.expand(B, -1, self.filter[1], -1)]
            weight_16 = [self.weight_hw_16.expand(B, -1, self.filter[2], -1), self.weight_wh_16.expand(B, -1, self.filter[2], -1)]
            weight_32 = [self.weight_hw_32.expand(B, -1, self.filter[3], -1), self.weight_wh_32.expand(B, -1, self.filter[3], -1)]
            flag = 0 if H == self.image_h3 else 1
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.mamba2(p1, weight_4, flag)
        p2 = self.pool2(c2)
        c3 = self.mamba3(p2, weight_8, flag)
        p3 = self.pool3(c3)
        c4 = self.mamba4(p3, weight_16, flag)
        p4 = self.pool4(c4)
        c5 = self.mamba5(p4, weight_32, flag)
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
        up_1_0 = self.upsample_1_0(fusion3_out)
        out = nn.Sigmoid()(up_1_0)
        return out
