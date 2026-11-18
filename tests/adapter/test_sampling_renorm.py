"""Run GPU cases with HCU_TEST_RENORM_GPU=1 in an adapter-enabled process."""

import os
from unittest.mock import Mock, patch
from types import ModuleType

import pytest
import torch

# Initialize adapter stubs before directly importing their flashinfer facade.
import sgl_kernel

from hcu_adapter.flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs


@pytest.mark.parametrize("kind", ["k", "p"])
def test_forwarding(kind):
    module_name = "sglang.kernels.ops.sampling.renorm_triton"
    module = ModuleType(module_name)
    impl = Mock(return_value=object())
    setattr(module, f"top_{kind}_renorm_probs_triton", impl)
    probs, threshold = object(), object()
    fn = top_k_renorm_probs if kind == "k" else top_p_renorm_probs
    with patch.dict("sys.modules", {module_name: module}):
        assert fn(probs, threshold) is impl.return_value
    impl.assert_called_once_with(probs, threshold)


@pytest.mark.parametrize("fn", [top_k_renorm_probs, top_p_renorm_probs])
def test_nan_check(fn):
    with pytest.raises(ValueError, match="NaN"):
        fn(torch.tensor([[float("nan"), 1.0]]), 1, check_nan=True)


@pytest.mark.skipif(os.environ.get("HCU_TEST_RENORM_GPU") != "1", reason="opt-in GPU")
@pytest.mark.parametrize("vocab", [32, 4096, 129280])
@pytest.mark.parametrize("kind", ["k", "p"])
@pytest.mark.parametrize("per_row", [False, True])
def test_public_interface_gpu(vocab, kind, per_row):
    from flashinfer import sampling
    from sglang.kernels.ops.sampling import renorm_triton
    from sglang.srt.speculative import dflash_utils

    fn = getattr(sampling, f"top_{kind}_renorm_probs")
    bridge = top_k_renorm_probs if kind == "k" else top_p_renorm_probs
    assert fn is bridge
    assert getattr(dflash_utils, f"top_{kind}_renorm_prob") is bridge
    torch.manual_seed(42)
    probs = torch.rand((6, vocab), device="cuda", dtype=torch.float32)
    probs[0].fill_(1)
    probs /= probs.sum(-1, keepdim=True)
    original = probs.clone()
    if kind == "p":
        threshold = torch.tensor([0.1, 0.5, 0.9, 0.95, 1.0, 0.8], device="cuda") if per_row else 0.95
        sorted_probs = probs.sort(dim=-1).values
        ps = torch.as_tensor(threshold, device="cuda").expand(6)
        cutoff = torch.searchsorted(sorted_probs.cumsum(-1), (1 - ps)[:, None]).clamp(max=vocab - 1)
        pivot = sorted_probs.gather(1, cutoff)
    else:
        threshold = torch.tensor([1, 2, 4, 8, 16, vocab], device="cuda") if per_row else 8
        ks = torch.as_tensor(threshold, device="cuda").expand(6)
        pivot = probs.sort(dim=-1, descending=True).values.gather(1, (ks - 1)[:, None])
    expected = torch.where(probs >= pivot, probs, 0)
    expected /= expected.sum(-1, keepdim=True)
    actual = fn(probs, threshold)
    direct = getattr(renorm_triton, f"top_{kind}_renorm_probs_triton")(probs, threshold)
    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(actual.sum(-1), torch.ones(6, device="cuda"), rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(probs, original, rtol=0, atol=0)
