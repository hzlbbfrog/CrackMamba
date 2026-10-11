"""Local selective scans on regular and shifted 2 x 2 patch grids."""

import time
import math
import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# Import the selective scan implementation from mamba_ssm.
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# Import an optional backend under separate names.
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


class SS2D_forward(nn.Module):
    """Apply a shared selective scan to independent image patches."""
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        branch=0,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.branch = branch
        self.d_model = d_model

        self.d_inner = int(0.5 * self.d_model)
        self.d_state = int(0.5 * self.d_model)

        self.dt_rank = int(self.d_model / 64) if dt_rank == "auto" else dt_rank

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=bias, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)  # Shape: [dt_rank + 2 * d_state, d_inner].
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)  # Shape: [d_inner, dt_rank].
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)  # Shape: [d_inner].
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # Shape: [d_inner, d_state].
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # Shape: [d_inner].

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Scale the time-step projection weights at initialization.
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize the bias so softplus produces time steps in [dt_min, dt_max].
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse softplus reference: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Mark the time-step bias for initialization routines that honor this flag.
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # Initialize the real-valued S4D state transition parameters.
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep the log transition parameters in float32.
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # Initialize the skip connection coefficients.
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep the skip connection coefficients in float32.
        D._no_weight_decay = True
        return D


    def forward(self, x, weight0):
        """Scan patch features with shape [B, N, C, P], where P is the patch area."""
        self.selective_scan = selective_scan_fn

        B, N, C, P = x.shape
        L = P
        K = N

        if self.branch == 1:
            xs = x
        else:
            xs = torch.flip(x, dims = [3])  # Reverse the flattened pixel order within each patch.
        x_dbl = torch.einsum("b n d l, c d -> b n c l", xs.view(B, N, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b n r l, d r -> b n d l", dts.view(B, N, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # Flatten patch groups and channels for the scan kernel.
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().unsqueeze(0).expand(N, -1).contiguous().view(-1)  # Repeat the shared skip coefficients across patches.
        A_logs = self.A_logs.float().unsqueeze(0).expand(N, -1, -1).contiguous().view(N*self.d_inner, self.d_state)  # Repeat the shared transition parameters across patches.
        As = -torch.exp(A_logs).contiguous()

        dt_projs_bias = self.dt_projs_bias.float().unsqueeze(0).expand(N, -1).contiguous().view(-1)  # Repeat the shared time-step bias across patches.

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        if self.branch == 1:
            y0_back = out_y
        else:
            y0_back = torch.flip(out_y, dims = [3])  # Restore the original pixel order within each patch.
        y = y0_back
        return y

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


class SS2D_local(nn.Module):
    """Fuse bidirectional scans over regular and shifted 2 x 2 patch grids."""
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model

        self.d_conv = d_conv
        self.d_inner = int(0.5 * self.d_model)

        self.in_proj_a1 = nn.Conv2d(self.d_model, self.d_inner, kernel_size=1, stride=1, padding=0, bias=False)

        self.in_proj_b1 = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=1, stride=1, padding=0, bias=False)

        self.out_proj = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_proj2 = nn.Conv2d(self.d_inner, self.d_model, kernel_size=1, stride=1, padding=0, bias=False)


        self.norm_BN_a = nn.BatchNorm2d(self.d_inner)
        self.norm_BN_b = nn.BatchNorm2d(self.d_inner)
        self.norm_BN_c = nn.BatchNorm2d(self.d_inner)

        self.norm_LN_a = LayerNorm(self.d_inner)
        self.norm_LN_b = LayerNorm(self.d_inner)

        self.act_a = nn.SiLU()
        self.act_c = nn.SiLU()
        self.act_b = nn.SiLU()

        self.ph = 2
        self.pw = 2

        self.forward_core_1 = SS2D_forward(d_model=d_model, d_state=d_state, branch=1)
        self.forward_core_2 = SS2D_forward(d_model=d_model, d_state=d_state, branch=2)

        self.forward_core_1_2 = SS2D_forward(d_model=d_model, d_state=d_state, branch=1)
        self.forward_core_2_2 = SS2D_forward(d_model=d_model, d_state=d_state, branch=2)


    def forward(self, x: torch.Tensor, weight0=0):

        """Return local scan features with the same [B, C, H, W] layout as the input."""
        B, C, H, W = x.shape

        a1 = self.in_proj_a1(x)
        a2 = self.act_a(self.norm_BN_a(a1))  # Apply batch normalization and SiLU to the projected features.

        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0  # Pad the bottom and right edges only when their sizes are odd.
        if H % 2 == 1:
            pad_top, pad_bottom = 0, 1
        if W % 2 == 1:
            pad_left, pad_right = 0, 1

        a2_padded = torch.nn.functional.pad(a2, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        new_H, new_W = H + pad_top + pad_bottom, W + pad_left + pad_right

        a3 = rearrange(a2_padded, 'b c (nh ph) (nw pw) -> b (nh nw) c (ph pw) ', ph=self.ph, pw=self.pw)  # Partition features into 2 x 2 patches: [B, N, C, 4].
        a3_1 = self.forward_core_1(a3, weight0)
        a3_2 = self.forward_core_2(a3, weight0)
        a4 = a3_1 + a3_2

        a4 = rearrange(a4, 'b (nh nw) c (ph pw) -> b c (nh ph) (nw pw)', nh=new_H//self.ph, nw=new_W//self.pw, ph=self.ph, pw=self.pw)
        a5 = a4[:, :, pad_top: H + pad_top, pad_left: W + pad_left]  # Remove padding to restore the original spatial resolution.
        a5 = self.norm_LN_a(a5)
        a6 = a2 * a5.sigmoid()

        # Scan a second patch grid shifted by one pixel.
        b1 = self.in_proj_b1(a6)
        b2 = self.act_b(self.norm_BN_b(b1))  # Apply batch normalization and SiLU to the projected features.

        # Pad the shifted grid so both spatial dimensions are divisible by two.
        if H % 2 == 1:
            pad_top, pad_bottom = 1, 0
        else:
            pad_top, pad_bottom = 1, 1
        if W % 2 == 1:
            pad_left, pad_right = 1, 0
        else:
            pad_left, pad_right = 1, 1

        b2_padded = torch.nn.functional.pad(b2, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        # Compute the padded spatial dimensions.
        new_H, new_W = H + pad_top + pad_bottom, W + pad_left + pad_right
        # Partition the shifted features into 2 x 2 patches.
        b3 = rearrange(b2_padded, 'b c (nh ph) (nw pw) -> b (nh nw) c (ph pw)', ph=self.ph, pw=self.pw)
        b3_1 = self.forward_core_1_2(b3, weight0)
        b3_2 = self.forward_core_2_2(b3, weight0)
        b4 = b3_1 + b3_2
        # Reconstruct the padded feature map.
        b4 = rearrange(b4, 'b (nh nw) c (ph pw) -> b c (nh ph) (nw pw)', nh=new_H//self.ph, nw=new_W//self.pw, ph=self.ph, pw=self.pw)
        # Remove padding to restore the original spatial resolution.
        b5 = b4[:, :, pad_top: H + pad_top, pad_left: W + pad_left]
        b5 = self.norm_LN_b(b5)

        b6 = b2 * b5.sigmoid()

        out = self.out_proj(b6)
        out = self.act_c(self.norm_BN_c(out))  # Apply batch normalization and SiLU to the output projection.
        out = self.out_proj2(out + a2 + b2)

        return  out
