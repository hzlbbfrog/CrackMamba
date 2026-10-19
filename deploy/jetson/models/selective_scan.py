"""Selective scan with a CUDA backend and an explicit portable recurrence.

The ONNX bridge emits a named custom placeholder which export_onnx.py replaces
with standard ONNX Loop operators. It never exports an opaque CUDA result as a
constant. The reference path is for correctness and portability, not paper FPS.
"""

import numpy as np
import torch
from torch.nn import functional as F


def reference_scan(u, delta, A, B, C, D, delta_bias):
    """Evaluate the real-valued, grouped Mamba selective recurrence in float32."""
    batch, channels, length = u.shape
    groups, states = B.shape[1:3]
    width = channels // groups
    values = u.float().reshape(batch, groups, width, length)
    steps = F.softplus(delta.float() + delta_bias.float()[None, :, None])
    steps = steps.reshape(batch, groups, width, length)
    decay = A.float().reshape(1, groups, width, states)
    state = u.new_zeros(batch, groups, width, states, dtype=torch.float32)
    result = []
    for index in range(length):
        dt = steps[..., index, None]
        value = values[..., index, None]
        state = torch.exp(dt * decay) * state + dt * B[:, :, None, :, index].float() * value
        result.append((state * C[:, :, None, :, index].float()).sum(-1))
    output = torch.stack(result, -1).reshape(batch, channels, length)
    return output + u.float() * D.float()[None, :, None]


def _numpy_reference(u, delta, A, B, C, D, delta_bias):
    """Supply real tracing values without recording thousands of inner operations."""
    arrays = [value.detach().float().cpu().numpy() for value in (u, delta, A, B, C, D, delta_bias)]
    values, steps, decay, source, readout, skip, bias = arrays
    batch, channels, length = values.shape
    groups, states = source.shape[1:3]
    width = channels // groups
    values = values.reshape(batch, groups, width, length)
    steps = np.logaddexp(np.float32(0), steps + bias[None, :, None])
    steps = steps.reshape(batch, groups, width, length)
    decay = decay.reshape(1, groups, width, states)
    state = np.zeros((batch, groups, width, states), dtype=np.float32)
    output = np.empty((batch, groups, width, length), dtype=np.float32)
    for index in range(length):
        dt = steps[..., index, None]
        state = np.exp(dt * decay) * state + dt * source[:, :, None, :, index] * values[..., index, None]
        output[..., index] = np.sum(state * readout[:, :, None, :, index], axis=-1)
    output = output.reshape(batch, channels, length) + arrays[0] * skip[None, :, None]
    return torch.from_numpy(output).to(u.device)


class _OnnxSelectiveScan(torch.autograd.Function):
    """Bridge the recurrence into an explicit node for subsequent Loop lowering."""

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias):
        return _numpy_reference(u, delta, A, B, C, D, delta_bias)

    @staticmethod
    def symbolic(graph, u, delta, A, B, C, D, delta_bias):
        batch, channels, length = u.type().sizes()
        groups, states = B.type().sizes()[1:3]
        if any(value is None for value in (batch, channels, length, groups, states)):
            raise RuntimeError("Selective-scan export requires fully specified static dimensions")
        result = graph.op("crackmamba::SelectiveScan", u, delta, A, B, C, D, delta_bias,
                          batch_i=batch, channels_i=channels, length_i=length,
                          groups_i=groups, states_i=states)
        return result.setType(u.type())


def selective_scan(u, delta, A, B, C, D, z=None, delta_bias=None,
                   delta_softplus=True, return_last_state=False, backend="mamba"):
    """Dispatch the exact inference contract, with no silent backend fallback."""
    if z is not None or not delta_softplus or return_last_state or delta_bias is None:
        raise ValueError("This deployment scan supports the trained model's real-valued inference contract only")
    if torch.onnx.is_in_onnx_export():
        return _OnnxSelectiveScan.apply(u, delta, A, B, C, D, delta_bias)
    if backend == "reference":
        return reference_scan(u, delta, A, B, C, D, delta_bias)
    if backend != "mamba":
        raise ValueError(f"Unknown selective scan backend: {backend}")
    if not u.is_cuda:
        raise RuntimeError("The Mamba backend requires CUDA; select --scan-backend reference on CPU")
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError as error:
        raise RuntimeError("Install the Jetson-compatible Mamba CUDA extension, or use the reference backend") from error
    return selective_scan_fn(u, delta, A, B, C, D, z=None, delta_bias=delta_bias,
                             delta_softplus=True, return_last_state=False)
