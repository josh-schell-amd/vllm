"""A/B for test_aiter_fused_moe_mi350_mxfp4_w4a16_accuracy under AITER v0.1.26.

Runs the test's exact case and scores the AITER output against two references
(silu*up, which the test uses, and gpt-oss SwiGLU, which CK-Tile's Swiglu
computes), with and without a zero w13/w2 bias. A zero bias does not change
the math but forces AITER's has_bias=True CK-Tile instance.

Run from the vLLM repo root on a gfx950 box:
    python debug_aiter_w4a16_swiglu.py
"""

import torch

import aiter
from tests.kernels.moe.test_rocm_aiter_moe import (
    _make_aiter_mxfp4_moe_case,
    ref_moe_forward,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    rocm_aiter_fused_experts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_quant_config,
)


def ref_swigluoai(x, w1, w2, topk_weights, topk_ids, alpha=1.702, limit=7.0):
    inter = w1.shape[1] // 2
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            gu = x[t].float() @ w1[e].float().T
            gate = gu[:inter].clamp(max=limit)
            up = gu[inter:].clamp(-limit, limit)
            act = (up + 1) * gate * torch.sigmoid(alpha * gate)
            out[t] += topk_weights[t, k] * (act @ w2[e].float().T)
    return out


def score(name, out, ref):
    diff = (out.float() - ref.float()).abs()
    within = (diff <= 0.1 + 0.1 * ref.float().abs()).float().mean().item()
    print(
        f"  vs {name:10s}: pass={within:7.2%} mean_abs={diff.mean().item():.4f} "
        f"max_abs={diff.max().item():.4f}"
    )


def main():
    print("aiter:", getattr(aiter, "__version__", "?"), aiter.__file__)
    case = _make_aiter_mxfp4_moe_case(
        num_tokens=32,
        hidden_dim=512,
        intermediate_dim=1024,
        num_experts=4,
        topk=2,
        seed=11,
    )
    print("w1.gate_up_interleave =", getattr(case.w1_kernel, "gate_up_interleave", None))
    refs = {
        "silu": ref_moe_forward(
            case.hidden_states, case.w1_ref, case.w2_ref, case.topk_weights, case.topk_ids
        ),
        "swigluoai": ref_swigluoai(
            case.hidden_states, case.w1_ref, case.w2_ref, case.topk_weights, case.topk_ids
        ),
    }

    num_experts, two_i, _ = case.w1_ref.shape
    hidden = case.w2_ref.shape[1]
    zero_bias_qc = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.AITER,
        w1_scale=case.quant_config.w1_scale,
        w2_scale=case.quant_config.w2_scale,
        w1_bias=torch.zeros(num_experts, two_i, dtype=torch.float32),
        w2_bias=torch.zeros(num_experts, hidden, dtype=torch.float32),
    )

    for label, qc in (("no bias (test as-is)", case.quant_config), ("zero bias", zero_bias_qc)):
        out = rocm_aiter_fused_experts(
            hidden_states=case.hidden_states,
            w1=case.w1_kernel,
            w2=case.w2_kernel,
            topk_weights=case.topk_weights,
            topk_ids=case.topk_ids,
            activation=MoEActivation.SWIGLUOAI,
            quant_config=qc,
            moe_config=case.moe_config,
            expert_mask=None,
        )
        torch.cuda.synchronize()
        print(f"{label}: finite={bool(torch.isfinite(out).all())}")
        for name, ref in refs.items():
            score(name, out, ref)


if __name__ == "__main__":
    main()
