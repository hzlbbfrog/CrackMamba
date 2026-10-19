"""Checkpoint-compatible global scans and explicit selective-scan backends."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .selective_scan import selective_scan


def rotate90(tensor, k):
    """Rotate NCHW spatial axes using ONNX-exportable flip/transpose operations."""
    turns = k % 4
    if turns == 1:
        return tensor.flip([3]).transpose(2, 3)
    if turns == 2:
        return tensor.flip([2, 3])
    if turns == 3:
        return tensor.flip([2]).transpose(2, 3)
    return tensor

class SS2D_forward(nn.Module):
    """Apply one group of four serpentine global scan directions."""

    def __init__(self, d_model, d_state=16, expand=2, dt_rank='auto', dt_min=0.001, dt_max=0.1, dt_init='random', dt_scale=1.0, dt_init_floor=0.0001, dropout=0.0, conv_bias=True, bias=False, device=None, dtype=None, branch=0, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.branch = branch
        self.d_model = d_model
        self.d_state = d_state * 4
        self.d_inner = int(1.0 * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 32) if dt_rank == 'auto' else dt_rank
        self.x_proj = (nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=bias, **factory_kwargs), nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=bias, **factory_kwargs), nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=bias, **factory_kwargs), nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=bias, **factory_kwargs))
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        self.dt_projs = (self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs), self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs), self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs), self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs))
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init='random', dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** (-0.5) * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def continuity(self, x, weight0):
        x_list = x.gather(dim=3, index=weight0)
        return x_list

    def continuity_back(self, x, weight0):
        x_out = x.gather(dim=3, index=weight0)
        return x_out

    def forward(self, x, weight, flag):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        if self.branch == 1:
            x0 = self.continuity(x.reshape(B, 1, C, L), weight[flag])
            x1 = self.continuity(rotate90(x, k=-3).reshape(B, 1, C, L), weight[1 - flag])
            x2 = self.continuity(rotate90(x, k=-2).reshape(B, 1, C, L), weight[flag])
            x3 = self.continuity(rotate90(x, k=-1).reshape(B, 1, C, L), weight[1 - flag])
        if self.branch == 7:
            x_flip_w = torch.flip(x, dims=[3])
            x0 = self.continuity(rotate90(x_flip_w, k=-3).reshape(B, 1, C, L), weight[1 - flag])
            x1 = self.continuity(rotate90(x_flip_w, k=-2).reshape(B, 1, C, L), weight[flag])
            x2 = self.continuity(rotate90(x_flip_w, k=-1).reshape(B, 1, C, L), weight[1 - flag])
            x3 = self.continuity(x_flip_w.reshape(B, 1, C, L), weight[flag])
        xs = torch.cat([x0, x1, x2, x3], dim=1)
        x_dbl = torch.einsum('b k d l, k c d -> b k c l', xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum('b k r l, k d r -> b k d l', dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False, backend=getattr(self, 'scan_backend', 'mamba')).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        if self.branch == 1:
            y0_back = self.continuity_back(out_y[:, 0].view(B, 1, C, L), weight[flag]).reshape(B, C, L).reshape(B, C, H, W)
            y1 = self.continuity_back(out_y[:, 1].view(B, 1, C, L), weight[1 - flag]).reshape(B, C, L).reshape(B, C, W, H)
            y1_back = rotate90(y1, k=3)
            y2 = self.continuity_back(out_y[:, 2].view(B, 1, C, L), weight[flag]).reshape(B, C, L).reshape(B, C, H, W)
            y2_back = rotate90(y2, k=2)
            y3 = self.continuity_back(out_y[:, 3].view(B, 1, C, L), weight[1 - flag]).reshape(B, C, L).reshape(B, C, W, H)
            y3_back = rotate90(y3, k=1)
        if self.branch == 7:
            y0 = self.continuity_back(out_y[:, 0].view(B, 1, C, L), weight[1 - flag]).reshape(B, C, L).reshape(B, C, W, H)
            y0_back = rotate90(y0, k=3)
            y0_back = torch.flip(y0_back, dims=[3])
            y1 = self.continuity_back(out_y[:, 1].view(B, 1, C, L), weight[flag]).reshape(B, C, L).reshape(B, C, H, W)
            y1_back = rotate90(y1, k=2)
            y1_back = torch.flip(y1_back, dims=[3])
            y2 = self.continuity_back(out_y[:, 2].view(B, 1, C, L), weight[1 - flag]).reshape(B, C, L).reshape(B, C, W, H)
            y2_back = rotate90(y2, k=1)
            y2_back = torch.flip(y2_back, dims=[3])
            y3 = self.continuity_back(out_y[:, 3].view(B, 1, C, L), weight[flag]).reshape(B, C, L).reshape(B, C, H, W)
            y3_back = torch.flip(y3, dims=[3])
        y = y0_back + y1_back + y2_back + y3_back
        return y

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

class SS2D_continous(nn.Module):
    """Two complementary global scans with attention-guided modulation."""

    def __init__(self, d_model, d_state=16, d_conv=3, conv_bias=True, bias=False, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.d_inner = int(1.0 * self.d_model)
        self.d_inner2 = int(0.1 * self.d_model)
        self.in_proj_a = nn.Conv2d(self.d_model, self.d_inner, kernel_size=1, stride=1, padding=0, bias=False)
        self.in_proj_b = nn.Conv2d(self.d_model, self.d_inner, kernel_size=1, stride=1, padding=0, bias=False)
        self.DW_conv2d_a = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner, bias=conv_bias, kernel_size=self.d_conv, padding=(self.d_conv - 1) // 2, **factory_kwargs)
        self.DW_conv2d_b = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner, bias=conv_bias, kernel_size=self.d_conv, padding=(self.d_conv - 1) // 2, **factory_kwargs)
        self.act_a = nn.SiLU()
        self.norm_BN_a = nn.BatchNorm2d(self.d_inner)
        self.norm_LN_a = LayerNorm(self.d_inner)
        self.act_b = nn.SiLU()
        self.norm_BN_b = nn.BatchNorm2d(self.d_inner)
        self.norm_LN_b = LayerNorm(self.d_inner)
        self.forward_core_1 = SS2D_forward(d_model=d_model, d_state=d_state, branch=1)
        self.forward_core_7 = SS2D_forward(d_model=d_model, d_state=d_state, branch=7)
        self.in_proj2 = nn.Conv2d(self.d_model, self.d_inner2, kernel_size=1, stride=1, padding=0, bias=False)
        self.act2 = nn.SiLU()
        self.out_proj2 = nn.Conv2d(self.d_inner2, self.d_inner, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor, weight0, flag):
        B, C, H, W = x.shape
        a1 = self.in_proj_a(x)
        a2 = self.act_a(self.norm_BN_a(self.DW_conv2d_a(a1)))
        a3 = self.forward_core_1(a2, weight0, flag)
        a3 = self.norm_LN_a(a3)
        a4 = a1 * a3.sigmoid()
        b1 = self.in_proj_b(a4)
        b2 = self.act_b(self.norm_BN_b(self.DW_conv2d_b(b1)))
        b3 = self.norm_LN_b(self.forward_core_7(b2, weight0, flag))
        b4 = b1 * b3.sigmoid()
        z0 = self.in_proj2(0.5 * a4 + b4)
        z1 = self.act2(z0)
        z1 = self.out_proj2(z1)
        out = 0.5 * a4 + b4 + z1
        return out
