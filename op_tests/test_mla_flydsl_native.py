# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native page64 FP8 MLA through the public FlyDSL stage-1/reduce wrappers.

Examples:
    python op_tests/test_mla_flydsl_native.py
    python op_tests/test_mla_flydsl_native.py -b 2 --layout h8 --kv-len 65 --splits 4

The default 96 small cases cover native H8/H64 and legacy H128 (both Q
strides), page tails, empty splits, and the five-stage LDS ring wraparound.
Every case checks eager execution, graph replay, and eager execution after a
different, long-context graph has populated LDS. Only the public wrapper pair
is timed; torch references, graph setup, poisoning and checks are not timed.
Importing this file does not select a device, import FlyDSL, or launch work.
"""

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ("gfx1250",)
PAGE_SIZE = 64
QK_DIM = 576
V_DIM = 512
SOFTMAX_SCALE = 192**-0.5
LAYOUTS = {
    "h8": (8, 576),
    "h64": (64, 576),
    "h128": (128, 576),
    "h128-stride768": (128, 768),
}
SCALES = {"nonunit": (0.75, 1.20), "unit": (1.0, 1.0)}
RTOL = ATOL = 0.15
TOL_ERR_RATIO = 0.01


class _GuardedTensor:
    """A native-sized contiguous view, not a padded-head kernel argument.

    The suffix can contain even an erroneous legacy H128 write. The prefix is
    256 elements to preserve the alignment needed by vector/TDM stores.
    Sentinels detect stores outside the view, not out-of-bounds reads.
    """

    def __init__(self, shape, dtype, suffix_elements, device):
        self.prefix = 256
        self.size = math.prod(shape)
        self.sentinel = 8192.0  # Exactly representable in both BF16 and FP32.
        self.storage = torch.full(
            (self.prefix + self.size + max(256, suffix_elements),),
            self.sentinel,
            dtype=dtype,
            device=device,
        )
        self.tensor = self.storage[self.prefix : self.prefix + self.size].view(shape)
        self.poison()

    def poison(self):
        self.tensor.fill_(float("nan"))

    def check(self, name):
        assert bool(
            (self.storage[: self.prefix] == self.sentinel).all()
        ), f"{name}: prefix guard overwritten"
        assert bool(
            (self.storage[self.prefix + self.size :] == self.sentinel).all()
        ), f"{name}: suffix guard overwritten"


def _make_case(batch, kv_len, heads, q_stride, splits, scales, seed):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((batch, heads, QK_DIM), device=device, generator=generator).to(
        torch.float8_e4m3fn
    )
    if q_stride != QK_DIM:
        backing = torch.full((batch, heads, q_stride), float("nan"), device=device).to(
            q.dtype
        )
        backing[..., :QK_DIM].copy_(q)
        q = backing[..., :QK_DIM]
    assert q.stride() == (heads * q_stride, q_stride, 1)
    if heads in (8, 64):
        assert q.is_contiguous()

    # Logical KV is used only by the reference. Each physical page is
    # [64 * 512 nope | 64 * 64 rope], NOT 64 interleaved 576-element rows.
    logical_kv = torch.randn(
        (batch, kv_len, QK_DIM), device=device, generator=generator
    ).to(q.dtype)
    pages_per_seq = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch * pages_per_seq
    pages = torch.full(
        (batch, pages_per_seq * PAGE_SIZE, QK_DIM), float("nan"), device=device
    ).to(q.dtype)
    pages[:, :kv_len].copy_(logical_kv)
    pages = pages.view(total_pages, PAGE_SIZE, QK_DIM)
    packed = torch.cat(
        (
            pages[..., :V_DIM].reshape(total_pages, -1),
            pages[..., V_DIM:].reshape(total_pages, -1),
        ),
        dim=-1,
    ).contiguous()
    # A non-identity page table also catches accidental physical-order reads.
    indices = torch.arange(total_pages - 1, -1, -1, device=device, dtype=torch.int32)
    # Byte views avoid relying on FP8 support in torch's flip implementation.
    kv_buffer = packed.view(torch.uint8).flip(0).contiguous().view(q.dtype)
    q_scale, kv_scale = (
        torch.tensor([value], device=device, dtype=torch.float32)
        for value in SCALES[scales]
    )
    guards = {
        "output": _GuardedTensor(
            (batch, heads, V_DIM), torch.bfloat16, batch * 128 * V_DIM, device
        ),
        "lse": _GuardedTensor(
            (batch, splits, heads, 1), torch.float32, batch * splits * 128, device
        ),
    }
    if splits > 1:
        guards["partials"] = _GuardedTensor(
            (batch, splits, heads, V_DIM),
            torch.float32,
            batch * splits * 128 * V_DIM,
            device,
        )
    return {
        "q": q,
        "logical_kv": logical_kv,
        "kv_buffer": kv_buffer,
        "kv_indptr": torch.arange(batch + 1, device=device, dtype=torch.int32)
        * pages_per_seq,
        "kv_page_indices": indices,
        "kv_last_page_lens": torch.full(
            (batch,), (kv_len - 1) % PAGE_SIZE + 1, device=device, dtype=torch.int32
        ),
        "qo_indptr": torch.arange(batch + 1, device=device, dtype=torch.int32),
        "num_kv_splits_indptr": torch.arange(
            batch + 1, device=device, dtype=torch.int32
        )
        * splits,
        "seqused_k": torch.full((batch,), kv_len, device=device, dtype=torch.int32),
        "q_scale": q_scale,
        "kv_scale": kv_scale,
        "splits": splits,
        "guards": guards,
        "output": guards["output"].tensor,
        "split_data": guards["output" if splits == 1 else "partials"].tensor,
        "split_lse": guards["lse"].tensor,
    }


def run_torch(q, logical_kv, q_scale, kv_scale, num_splits):
    """FP32 attention and round-robin page partials; final output cast to BF16.

    Empty splits are deliberately NaN, matching untouched stage-1 scratch.
    The public reducer ignores these slots using ceil(seqused_k / 64).
    """
    qf, kvf = q.to(torch.float32), logical_kv.to(torch.float32)
    logits = torch.matmul(qf, kvf.transpose(1, 2)) * (
        SOFTMAX_SCALE * q_scale * kv_scale
    )
    values = kvf[..., :V_DIM] * kv_scale
    output = torch.matmul(torch.softmax(logits, dim=-1), values).to(torch.bfloat16)
    batch, heads, kv_len = logits.shape
    partials = torch.full(
        (batch, num_splits, heads, V_DIM),
        float("nan"),
        device=q.device,
        dtype=torch.float32,
    )
    lse = torch.full(
        (batch, num_splits, heads, 1),
        float("nan"),
        device=q.device,
        dtype=torch.float32,
    )
    token_pages = torch.arange(kv_len, device=q.device) // PAGE_SIZE
    for split in range(min(num_splits, (kv_len + PAGE_SIZE - 1) // PAGE_SIZE)):
        mask = token_pages % num_splits == split
        split_logits = logits[..., mask]
        partials[:, split] = torch.matmul(
            torch.softmax(split_logits, dim=-1), values[:, mask]
        )
        lse[:, split, :, 0] = torch.logsumexp(split_logits, dim=-1)
    return output, partials, lse


def _launcher(case):
    # Lazy import: neither test collection nor the architecture gate needs FlyDSL.
    from aiter.ops.flydsl.mla_kernels import (
        flydsl_mla_decode_reduce,
        flydsl_mla_pagesize64_fp8_fp8,
    )

    def launch():
        flydsl_mla_pagesize64_fp8_fp8(
            case["split_data"],
            case["split_lse"],
            case["q"],
            case["kv_buffer"],
            case["kv_indptr"],
            case["kv_page_indices"],
            case["kv_last_page_lens"],
            case["qo_indptr"],
            case["num_kv_splits_indptr"],
            case["q_scale"],
            case["kv_scale"],
            SOFTMAX_SCALE,
            case["splits"],
            page_size=PAGE_SIZE,
        )
        if case["splits"] > 1:
            flydsl_mla_decode_reduce(
                case["split_data"],
                case["split_lse"],
                case["seqused_k"],
                case["output"],
                case["splits"],
                1,
            )
        return case["output"]

    return launch


def _check_close(reference, actual, name):
    assert bool(torch.isfinite(reference).all()), f"{name}: non-finite reference"
    assert bool(torch.isfinite(actual).all()), f"{name}: non-finite result"
    err = checkAllclose(
        reference.to(torch.float32),
        actual.to(torch.float32),
        rtol=RTOL,
        atol=ATOL,
        tol_err_ratio=TOL_ERR_RATIO,
        msg=name,
    )
    # checkAllclose logs and returns the mismatch ratio; it need not raise.
    assert err <= TOL_ERR_RATIO, f"{name}: mismatch ratio {err}"
    return err


def _check_case(case, reference, name):
    out_ref, partial_ref, lse_ref = reference
    for key, guard in case["guards"].items():
        guard.check(f"{name} {key}")
    err = _check_close(out_ref, case["output"], f"{name} output")
    valid = torch.isfinite(lse_ref)
    _check_close(lse_ref[valid], case["split_lse"][valid], f"{name} LSE")
    # The kernel does not initialize empty splits; the reducer must not read
    # them. Poison, rather than zero, makes accidental consumption observable.
    assert bool(
        torch.isnan(case["split_lse"][~valid]).all()
    ), f"{name}: empty-split LSE was written"
    if case["splits"] > 1:
        valid_data = valid.expand_as(partial_ref)
        _check_close(
            partial_ref[valid_data],
            case["split_data"][valid_data],
            f"{name} FP32 partials",
        )
        assert bool(
            torch.isnan(case["split_data"][~valid_data]).all()
        ), f"{name}: empty-split partials were written"
    return err


def _poison(case):
    for guard in case["guards"].values():
        guard.poison()


def _capture(launch):
    # Compile and warm up on a side stream, outside capture. All allocation,
    # host-visible assertions and reference work likewise stay outside capture.
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        launch()
        launch()
    current.wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    current.wait_stream(stream)
    return graph


@benchmark()
def test_mla_flydsl_native(batch, kv_len, layout, splits, scales):
    if batch < 1 or kv_len < 1 or splits < 1:
        raise ValueError("batch, kv_len and splits must be positive")
    heads, q_stride = LAYOUTS[layout]
    case = _make_case(batch, kv_len, heads, q_stride, splits, scales, seed=20260909)
    reference = run_torch(
        case["q"], case["logical_kv"], case["q_scale"], case["kv_scale"], splits
    )
    # A different full-head, long-context direct-output graph leaves LDS holding
    # other values (including BF16 output staging). Replay immediately before
    # the target, with no intervening torch kernel to obscure stale-LDS bugs.
    primer = _make_case(2, 1281, 64, 576, 1, "nonunit", seed=20260910)
    primer_graph = _capture(_launcher(primer))
    candidates = {"flydsl": _launcher(case)}
    flops = 2 * batch * heads * kv_len * (QK_DIM + V_DIM)
    # Logical traffic: Q/KV once, metadata, final output, plus stage-1 writes
    # and reducer reads of valid FP32 partials/LSE. This is not measured HBM
    # traffic; head-group reloads and cache effects are not counted.
    valid_splits = min(splits, (kv_len + PAGE_SIZE - 1) // PAGE_SIZE)
    nbytes = (
        case["q"].numel() * case["q"].element_size()
        + case["logical_kv"].numel() * case["logical_kv"].element_size()
        + case["output"].numel() * case["output"].element_size()
        + sum(
            case[key].numel() * case[key].element_size()
            for key in (
                "kv_indptr",
                "kv_page_indices",
                "kv_last_page_lens",
                "qo_indptr",
                "num_kv_splits_indptr",
                "q_scale",
                "kv_scale",
            )
        )
        + batch * valid_splits * heads * 4  # LSE writes, including S1.
    )
    if splits > 1:
        nbytes += batch * valid_splits * heads * (2 * V_DIM + 1) * 4
        nbytes += case["seqused_k"].numel() * case["seqused_k"].element_size()
    ret = {"gfx": get_gfx()}
    for name, launch in candidates.items():
        launch()
        errors = [_check_case(case, reference, f"{name} eager")]
        graph = _capture(launch)
        for mode in ("eager-after-graph", "graph-replay", "graph-replay-repeat"):
            _poison(case)
            primer_graph.replay()
            if mode == "eager-after-graph":
                launch()
            else:
                graph.replay()
            errors.append(_check_case(case, reference, f"{name} {mode}"))
        for key, guard in primer["guards"].items():
            guard.check(f"primer {key}")
        _poison(case)
        _, us = run_perftest(
            launch, num_iters=21, num_warmup=2, num_rotate_args=1, use_cuda_event=True
        )
        errors.append(_check_case(case, reference, f"{name} timed"))
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(errors)
    return ret


# This is a standalone benchmark with explicit axes, not pytest fixtures.
# Importers can still call the decorated function directly.
test_mla_flydsl_native.__test__ = False


def main():
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning(
            "Native page64 FlyDSL MLA unsupported on %s; skipping", gfx
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument("-b", "--batch", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--kv-len", type=int, nargs="+", default=[1, 63, 64, 65, 1089, 1281]
    )
    parser.add_argument(
        "--layout", choices=tuple(LAYOUTS), nargs="+", default=list(LAYOUTS)
    )
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--scales", choices=tuple(SCALES), nargs="+", default=["nonunit"]
    )
    args = parser.parse_args()
    if any(value < 1 for value in (*args.batch, *args.kv_len, *args.splits)):
        parser.error("batch, kv-len and splits must be positive")
    rows = []
    for batch, kv_len, layout, splits, scales in itertools.product(
        args.batch, args.kv_len, args.layout, args.splits, args.scales
    ):
        rows.append(test_mla_flydsl_native(batch, kv_len, layout, splits, scales))
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "Native page64 FlyDSL MLA summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
