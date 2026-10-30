# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .kernels.tensor_shim import _run_compiled, ptr_arg

__all__ = [
    "flydsl_mla_decode_reduce",
    "flydsl_mla_pagesize1_bf16_bf16",
    "flydsl_mla_pagesize1_fp8_fp8",
    "flydsl_mla_pagesize64_fp8_fp8",
]

_PAGESIZE1_NUM_Q_HEADS = (16, 32, 64, 96, 128)
_PAGESIZE1_PACKED_Q_HEADS = 64


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _require_runtime(condition, message):
    if not condition:
        raise RuntimeError(message)


def _require_layout(name, tensor, dtype, shape=None):
    """Check the dtype, shape and contiguity a kernel argument must satisfy.

    ``shape`` entries may be ``None`` to leave a dimension free.
    """
    _require(tensor.dtype == dtype, f"{name}: expected {dtype}, got {tensor.dtype}")
    if shape is not None:
        actual = tuple(tensor.shape)
        ok = len(actual) == len(shape) and all(
            want is None or got == want for got, want in zip(actual, shape)
        )
        _require(ok, f"{name}: expected shape {list(shape)}, got {list(actual)}")
    _require(tensor.is_contiguous(), f"{name}: expected a contiguous tensor")


def _validate_pagesize1_inputs(
    split_data,
    split_lse,
    final_output,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    q_scale,
    kv_scale,
    final_lse,
    max_seqlen_q,
    causal,
    qkv_dtype,
):
    arch = str(get_rocm_arch() or "").split(":", 1)[0]
    _require_runtime(
        arch == "gfx1250",
        f"expected gfx1250, got {arch or 'unknown'}",
    )
    softmax_scale = float(softmax_scale)
    max_seqlen_q = int(max_seqlen_q)
    total_q = q.size(0)
    num_q_heads = q.size(1)
    _require(
        num_q_heads in _PAGESIZE1_NUM_Q_HEADS,
        f"q: expected one of {list(_PAGESIZE1_NUM_Q_HEADS)} heads, "
        f"got {num_q_heads}",
    )
    _require(
        max_seqlen_q in (1, 2, 3, 4) or num_q_heads > _PAGESIZE1_PACKED_Q_HEADS,
        f"max_seqlen_q: expected one of [1, 2, 3, 4], got {max_seqlen_q}",
    )
    _require(
        num_q_heads > _PAGESIZE1_PACKED_Q_HEADS
        or num_q_heads * max_seqlen_q <= 2 * _PAGESIZE1_PACKED_Q_HEADS,
        f"q: {num_q_heads} heads support max_seqlen_q up to "
        f"{min(4, 2 * _PAGESIZE1_PACKED_Q_HEADS // num_q_heads)}, got {max_seqlen_q}",
    )
    _require_layout("q", q, qkv_dtype, (total_q, num_q_heads, 576))
    _require_layout("kv_buffer", kv_buffer, qkv_dtype, (None, 1, 1, 576))
    _require_layout("split_data", split_data, torch.float32, (None, num_q_heads, 512))
    _require_layout(
        "split_lse", split_lse, torch.float32, (split_data.size(0), num_q_heads)
    )
    _require_layout(
        "final_output", final_output, torch.bfloat16, (total_q, num_q_heads, 512)
    )
    _require_layout("kv_page_indices", kv_page_indices, torch.int32)
    _require_layout("work_indptr", work_indptr, torch.int32)
    _require_layout("work_info", work_info, torch.int32, (None, 8))
    if qkv_dtype == torch.float8_e4m3fn:
        for name, scale in (("q_scale", q_scale), ("kv_scale", kv_scale)):
            _require(scale is not None, f"{name}: expected a float32 scalar tensor")
    # Only the work items the planner left un-split write here; the ones it did
    # split are the reduce's job. None means the caller wants no LSE at all.
    if final_lse is not None:
        _require_layout("final_lse", final_lse, torch.float32, (total_q, num_q_heads))

    properties = torch.cuda.get_device_properties(q.device)
    lds_size = getattr(properties, "shared_memory_per_multiprocessor", None)
    lds_size = int(lds_size) if lds_size is not None else get_lds_capacity_bytes(arch)
    return (
        work_indptr.numel() - 1,
        lds_size,
        softmax_scale,
        num_q_heads,
        max_seqlen_q,
        int(bool(causal)),
        int(final_lse is not None),
    )


def _run_pagesize1(
    launch,
    qkv_dtype,
    qkv_type,
    split_data,
    split_lse,
    final_output,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    q_scale,
    kv_scale,
    final_lse,
    max_seqlen_q,
    causal,
    qo_indptr,
    kv_indptr,
    g_kv_indptr,
    cp_world_size,
    cp_rank,
    stream,
):
    (
        num_cus,
        lds_size,
        softmax_scale,
        num_q_heads,
        max_seqlen_q,
        causal,
        write_final_lse,
    ) = _validate_pagesize1_inputs(
        split_data,
        split_lse,
        final_output,
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info,
        softmax_scale,
        q_scale,
        kv_scale,
        final_lse,
        max_seqlen_q,
        causal,
        qkv_dtype,
    )
    cp_world_size = int(cp_world_size)
    cp_rank = int(cp_rank)
    _require(cp_world_size >= 1, f"cp_world_size: expected >= 1, got {cp_world_size}")
    cp_round_robin = int(cp_world_size > 1 and bool(causal))
    if cp_round_robin:
        _require(
            0 <= cp_rank < cp_world_size,
            f"cp_rank: expected in [0, {cp_world_size}), got {cp_rank}",
        )
        _require(
            qo_indptr is not None and kv_indptr is not None and g_kv_indptr is not None,
            "round-robin CP needs qo_indptr, kv_indptr and g_kv_indptr",
        )
        _require_layout("qo_indptr", qo_indptr, torch.int32, (None,))
        _require_layout("kv_indptr", kv_indptr, torch.int32, (qo_indptr.numel(),))
        _require_layout("g_kv_indptr", g_kv_indptr, torch.int32, (qo_indptr.numel(),))
    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    buffers = (
        ptr_arg(split_data, fx.Float32),
        ptr_arg(split_lse, fx.Float32),
        ptr_arg(final_output, fx.BFloat16),
        (
            flyc.from_c_void_p(fx.Float32, 0)
            if final_lse is None
            else ptr_arg(final_lse, fx.Float32)
        ),
        ptr_arg(q, qkv_type),
        ptr_arg(kv_buffer, qkv_type),
        ptr_arg(kv_page_indices, fx.Int32),
        ptr_arg(work_indptr, fx.Int32),
        ptr_arg(work_info, fx.Int32),
        *(
            (
                flyc.from_c_void_p(fx.Float32, 0)
                if scale is None
                else ptr_arg(scale, fx.Float32)
            )
            for scale in (q_scale, kv_scale)
        ),
        softmax_scale,
        kv_buffer.size(0),
        kv_page_indices.numel(),
        (
            ptr_arg(qo_indptr, fx.Int32)
            if cp_round_robin
            else flyc.from_c_void_p(fx.Int32, 0)
        ),
        (
            ptr_arg(kv_indptr, fx.Int32)
            if cp_round_robin
            else flyc.from_c_void_p(fx.Int32, 0)
        ),
        (
            ptr_arg(g_kv_indptr, fx.Int32)
            if cp_round_robin
            else flyc.from_c_void_p(fx.Int32, 0)
        ),
        cp_world_size if cp_round_robin else 1,
        cp_rank if cp_round_robin else 0,
    )
    _run_compiled(
        launch,
        *buffers,
        num_q_heads,
        max_seqlen_q,
        causal,
        cp_round_robin,
        write_final_lse,
        num_cus,
        lds_size,
        stream,
        specialization_key=(
            num_q_heads,
            max_seqlen_q,
            causal,
            cp_round_robin,
            write_final_lse,
            num_cus,
            lds_size,
        ),
    )


def flydsl_mla_pagesize1_fp8_fp8(
    split_data,
    split_lse,
    final_output,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    *,
    q_scale,
    kv_scale,
    final_lse=None,
    max_seqlen_q=1,
    causal=False,
    qo_indptr=None,
    kv_indptr=None,
    g_kv_indptr=None,
    cp_world_size=1,
    cp_rank=0,
    stream=None,
):
    from .kernels.mla_gfx1250.mla_pagesize1_fp8_fp8 import (
        launch_mla_pagesize1_fp8_fp8,
    )

    _run_pagesize1(
        launch_mla_pagesize1_fp8_fp8,
        torch.float8_e4m3fn,
        fx.Int8,
        split_data,
        split_lse,
        final_output,
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info,
        softmax_scale,
        q_scale,
        kv_scale,
        final_lse,
        max_seqlen_q,
        causal,
        qo_indptr,
        kv_indptr,
        g_kv_indptr,
        cp_world_size,
        cp_rank,
        stream,
    )


def flydsl_mla_pagesize1_bf16_bf16(
    split_data,
    split_lse,
    final_output,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    *,
    final_lse=None,
    max_seqlen_q=1,
    causal=False,
    qo_indptr=None,
    kv_indptr=None,
    g_kv_indptr=None,
    cp_world_size=1,
    cp_rank=0,
    stream=None,
):
    """BF16 Q/KV form of :func:`flydsl_mla_pagesize1_fp8_fp8`; no scales."""
    from .kernels.mla_gfx1250.mla_pagesize1_bf16_bf16 import (
        launch_mla_pagesize1_bf16_bf16,
    )

    _run_pagesize1(
        launch_mla_pagesize1_bf16_bf16,
        torch.bfloat16,
        fx.BFloat16,
        split_data,
        split_lse,
        final_output,
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info,
        softmax_scale,
        None,
        None,
        final_lse,
        max_seqlen_q,
        causal,
        qo_indptr,
        kv_indptr,
        g_kv_indptr,
        cp_world_size,
        cp_rank,
        stream,
    )


def _validate_pagesize64_inputs(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    qo_indptr,
    num_kv_splits_indptr,
    q_scale,
    kv_scale,
    num_splits,
    page_size,
):
    fp8 = torch.float8_e4m3fn
    _require_runtime(
        str(get_rocm_arch()).split(":", 1)[0] == "gfx1250", "expected gfx1250"
    )
    _require(page_size == 64, "page_size must be 64")
    _require(
        isinstance(num_splits, int) and num_splits > 0, "num_splits must be positive"
    )
    _require(
        q.ndim == 3 and q.shape[1] in (8, 64, 128) and q.shape[2] == 576,
        "q must have shape [batch, 8/64/128, 576]",
    )
    batch, heads, _ = q.shape
    _require(batch > 0, "q must contain at least one sequence")
    _require(q.dtype == fp8, "q must use float8_e4m3fn")
    valid_strides = (576, 768) if heads == 128 else (576,)
    _require(
        q.stride(2) == 1
        and q.stride(1) in valid_strides
        and q.stride(0) == heads * q.stride(1),
        "unsupported q strides",
    )
    for tensor in (
        split_data,
        split_lse,
        kv_buffer,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        qo_indptr,
        num_kv_splits_indptr,
        q_scale,
        kv_scale,
    ):
        _require(tensor.device == q.device, "all tensors must be on q.device")
    _require(q.is_cuda, "q must be on a GPU")
    _require_layout("kv_buffer", kv_buffer, fp8, (None, 64 * 576))
    for name, tensor, shape in (
        ("kv_indptr", kv_indptr, (batch + 1,)),
        ("kv_page_indices", kv_page_indices, (None,)),
        ("kv_last_page_lens", kv_last_page_lens, (batch,)),
        ("qo_indptr", qo_indptr, (batch + 1,)),
        ("num_kv_splits_indptr", num_kv_splits_indptr, (batch + 1,)),
        ("q_scale", q_scale, (1,)),
        ("kv_scale", kv_scale, (1,)),
    ):
        dtype = torch.float32 if name.endswith("scale") else torch.int32
        _require_layout(name, tensor, dtype, shape)

    if num_splits == 1:
        _require_layout("split_data", split_data, torch.bfloat16, (batch, heads, 512))
    else:
        _require_layout(
            "split_data", split_data, torch.float32, (batch, num_splits, heads, 512)
        )
    _require_layout(
        "split_lse", split_lse, torch.float32, (batch, num_splits, heads, 1)
    )
    return batch


def flydsl_mla_pagesize64_fp8_fp8(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    qo_indptr,
    num_kv_splits_indptr,
    q_scale,
    kv_scale,
    softmax_scale,
    num_splits,
    *,
    page_size=64,
    stream=None,
):
    """Decode one query per sequence from packed FP8 page64 KV on gfx1250.

    Q is native [batch, heads, 576] for heads 8/64/128. Legacy H128 Q with
    head stride 768 is also supported. KV pages contain contiguous [64,512]
    NoPE followed by [64,64] RoPE, rather than interleaved token rows.
    S1 writes BF16 [batch,heads,512]; otherwise FP32 [batch,splits,heads,512]
    partials are merged by ``flydsl_mla_decode_reduce``. LSE is always FP32
    [batch,splits,heads,1]. Empty splits remain untouched.

    The caller supplies valid page indices, nonempty sequences, one query per
    sequence in qo_indptr, and last-page lengths in [1,64]. Metadata contents
    are not copied to the host, so this wrapper remains graph-capturable.
    """
    batch = _validate_pagesize64_inputs(
        split_data,
        split_lse,
        q,
        kv_buffer,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        qo_indptr,
        num_kv_splits_indptr,
        q_scale,
        kv_scale,
        num_splits,
        page_size,
    )
    from .kernels.mla_gfx1250.mla_pagesize64_fp8_fp8 import (
        launch_mla_pagesize64_fp8_fp8,
    )

    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    output_type = fx.BFloat16 if num_splits == 1 else fx.Float32
    _run_compiled(
        launch_mla_pagesize64_fp8_fp8,
        ptr_arg(split_data, output_type),
        ptr_arg(split_lse, fx.Float32),
        ptr_arg(q, fx.Int8),
        ptr_arg(kv_buffer, fx.Int8),
        ptr_arg(kv_indptr, fx.Int32),
        ptr_arg(kv_page_indices, fx.Int32),
        ptr_arg(kv_last_page_lens, fx.Int32),
        ptr_arg(qo_indptr, fx.Int32),
        ptr_arg(num_kv_splits_indptr, fx.Int32),
        ptr_arg(q_scale, fx.Float32),
        ptr_arg(kv_scale, fx.Float32),
        float(softmax_scale),
        batch,
        num_splits,
        int(num_splits == 1),
        stream,
        q.shape[1],
        q.stride(1),
        specialization_key=(int(num_splits == 1), q.shape[1], q.stride(1)),
    )


def flydsl_mla_decode_reduce(
    split_data,  # fp32 [total_tokens, num_splits, num_heads, v_head_dim]
    split_lse,  # fp32 [total_tokens, num_splits, num_heads, 1]
    seqused_k,  # int32 [batch]
    out,  # [total_tokens, num_heads, v_head_dim]
    num_splits,
    num_tokens_per_seq,
    stream=None,
):
    """Merge the per-split partials of :func:`flydsl_mla_decode_fwd` into ``out``.

    Split out as its own entry point so a caller that ran stage 1 with
    ``skip_reduce`` can drive the merge itself, and so the two stages can be
    timed independently.
    """
    from .kernels.mla_reduce import compile_mla_decode_reduce

    total_tokens, _, num_heads, v_head_dim = split_data.shape
    out_dtype = "bf16" if out.dtype == torch.bfloat16 else "fp16"
    launch = compile_mla_decode_reduce(H=num_heads, Dv=v_head_dim, out_dtype=out_dtype)
    if stream is None:
        stream = torch.cuda.current_stream(out.device)
    out_type = fx.BFloat16 if out_dtype == "bf16" else fx.Float16
    _run_compiled(
        launch,
        ptr_arg(split_data, fx.Float32),
        ptr_arg(split_lse, fx.Float32),
        ptr_arg(seqused_k, fx.Int32),
        ptr_arg(out, out_type),
        total_tokens,
        seqused_k.numel(),
        num_splits,
        num_tokens_per_seq,
        stream,
    )
    return out
