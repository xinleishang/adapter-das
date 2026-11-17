# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public bridge contract against the vendor op, not Python-reference parity."""

from types import SimpleNamespace

import pytest
import torch

import sgl_kernel
import sgl_kernel.elementwise as elementwise


def _has_hip_kernel():
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "sgl_kernel::rope_fp4_fake_quant", "CUDA"
        )
    except RuntimeError:
        return False


requires_vendor = pytest.mark.skipif(
    not _has_hip_kernel(), reason="requires the vendor RoPE/FP4 HIP kernel"
)


def test_public_export():
    assert sgl_kernel.rope_fp4_fake_quant is elementwise.rope_fp4_fake_quant
    assert "rope_fp4_fake_quant" in sgl_kernel.__all__


@pytest.mark.parametrize("compressed_kv", [False, True])
def test_forwards_arguments(monkeypatch, compressed_kv):
    x, freqs, output = object(), object(), object()
    calls = []

    def fake_op(*args):
        calls.append(args)
        return output

    fake_torch = SimpleNamespace(ops=SimpleNamespace(sgl_kernel=SimpleNamespace(
        rope_fp4_fake_quant=SimpleNamespace(default=fake_op)
    )))
    monkeypatch.setattr(elementwise, "torch", fake_torch)
    assert elementwise.rope_fp4_fake_quant(x, freqs, 64, compressed_kv) is output
    assert calls == [(x, freqs, 64, compressed_kv)]


def test_missing_vendor_op_has_actionable_error(monkeypatch):
    fake_torch = SimpleNamespace(ops=SimpleNamespace(sgl_kernel=SimpleNamespace()))
    monkeypatch.setattr(elementwise, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="requires a vendor sgl_kernel build"):
        elementwise.rope_fp4_fake_quant(object(), object(), 64)


def test_does_not_swallow_vendor_errors(monkeypatch):
    error = RuntimeError("vendor execution failed")

    def fake_op(*args):
        raise error

    fake_torch = SimpleNamespace(ops=SimpleNamespace(sgl_kernel=SimpleNamespace(
        rope_fp4_fake_quant=SimpleNamespace(default=fake_op)
    )))
    monkeypatch.setattr(elementwise, "torch", fake_torch)
    with pytest.raises(RuntimeError) as caught:
        elementwise.rope_fp4_fake_quant(object(), object(), 64)
    assert caught.value is error


def _inputs(shape, seed):
    torch.manual_seed(seed)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn((shape[0], 32), device="cuda", dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles).contiguous()
    return x, freqs


@requires_vendor
@pytest.mark.parametrize("shape", [
    (1, 128), (17, 32, 128), (193, 512),
    (16384, 32, 128), (32768, 32, 128),
])
@pytest.mark.parametrize("compressed_kv", [False, True])
def test_matches_vendor(shape, compressed_kv):
    x, freqs = _inputs(shape, 20261116)
    original = x.clone()
    expected = torch.ops.sgl_kernel.rope_fp4_fake_quant.default(
        x, freqs, 64, compressed_kv
    )
    actual = sgl_kernel.rope_fp4_fake_quant(x, freqs, 64, compressed_kv)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
    assert torch.equal(x, original)
    assert actual.shape == x.shape and actual.dtype == x.dtype


@requires_vendor
@pytest.mark.parametrize("tokens", [1, 17, 16384])
@pytest.mark.parametrize("compressed_kv", [False, True])
def test_graph_replays_match_vendor(tokens, compressed_kv):
    x, freqs = _inputs((tokens, 32, 128), 47)
    for _ in range(3):
        output = sgl_kernel.rope_fp4_fake_quant(x, freqs, 64, compressed_kv)
    del output
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = sgl_kernel.rope_fp4_fake_quant(x, freqs, 64, compressed_kv)
    for seed in (100, 101, 102):
        updated, rotations = _inputs(x.shape, seed)
        x.copy_(updated)
        freqs.copy_(rotations)
        graph.replay()
        expected = torch.ops.sgl_kernel.rope_fp4_fake_quant.default(
            x, freqs, 64, compressed_kv
        )
        torch.cuda.synchronize()
        assert torch.equal(output, expected)
