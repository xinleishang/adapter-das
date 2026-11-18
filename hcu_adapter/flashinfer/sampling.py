"""Lazy bridges to SGLang's HIP probability-renormalization kernels."""


def _check_probs(probs, check_nan):
    if check_nan and probs.isnan().any().item():
        raise ValueError("Input probabilities contain NaN.")


def top_p_renorm_probs(probs, top_p, check_nan=False):
    _check_probs(probs, check_nan)
    from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

    return top_p_renorm_probs_triton(probs, top_p)


def top_k_renorm_probs(probs, top_k, check_nan=False):
    _check_probs(probs, check_nan)
    from sglang.kernels.ops.sampling.renorm_triton import top_k_renorm_probs_triton

    return top_k_renorm_probs_triton(probs, top_k)
