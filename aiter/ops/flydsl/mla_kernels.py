# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .kernels.tensor_shim import _run_compiled, ptr_arg

__all__ = [
    "MLADecodePagesize64Workspace",
    "flydsl_mla_decode_pagesize64_fp8",
    "flydsl_mla_decode_pagesize64_fp8_workspace",
    "flydsl_mla_decode_reduce",
    "flydsl_mla_pagesize1_bf16_bf16",
    "flydsl_mla_pagesize1_fp8_fp8",
    "flydsl_mla_pagesize64_fp8_fp8",
    "flydsl_mla_pagesize64_fp8_fp8_ring",
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
        arch in ("gfx1250", "gfx1250-strict"),
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
    compact_ring=False,
    kv_num_stages=5,
    q_from_lds=False,
    waves_per_eu=0,
    pv_load_depth=8,
    q_load_depth=0,
    q_lds_pitch=576,
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
    _require(kv_num_stages in ((3, 4) if compact_ring else (5,)), "unsupported KV stage count for ring variant")
    _require(compact_ring or not q_from_lds, "Q LDS requires compact_ring")
    _require(waves_per_eu in (0, 1, 2), "waves_per_eu must be 0, 1 or 2")
    _require(compact_ring or waves_per_eu == 0, "occupancy hint requires compact_ring")
    _require(q_lds_pitch in (576, 592), "unsupported Q LDS pitch")
    _require(q_from_lds or q_lds_pitch == 576, "Q LDS pitch requires Q LDS")
    _require(pv_load_depth in (4, 8, 12, 16), "unsupported PV load depth")
    _require(q_load_depth in (0, 1, 2, 4), "unsupported Q load depth")
    _require(q_from_lds or q_load_depth == 0, "Q load depth requires Q LDS")
    _require(compact_ring or (pv_load_depth == 8 and q_load_depth == 0), "load tuning requires compact_ring")
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
    from flydsl.compiler.kernel_function import CompilationContext

    with CompilationContext.compile_hints({"waves_per_eu": waves_per_eu}):
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
            int(bool(compact_ring)),
            int(kv_num_stages),
            int(bool(q_from_lds)),
            int(pv_load_depth),
            int(q_load_depth),
            int(q_lds_pitch),
            specialization_key=(int(num_splits == 1), q.shape[1], q.stride(1), int(bool(compact_ring)), int(kv_num_stages), int(bool(q_from_lds)), int(waves_per_eu), int(pv_load_depth), int(q_load_depth), int(q_lds_pitch)),
        )



def flydsl_mla_pagesize64_fp8_fp8_ring(
    split_data, split_lse, q, kv, block_tables, seqused_k,
    q_scale, kv_scale, softmax_scale, num_splits, *,
    kv_num_stages=3, waves_per_eu=1, pipelined=False, pv_load_depth=1,
    partial_bf16=False, stream=None,
):
    """Launch native H64 page64 MLA without packing Q or KV.

    S1 writes BF16 output. Split launches write normalized partials (FP32, or
    BF16 when ``partial_bf16``) and FP32 natural-log LSE for
    ``flydsl_mla_decode_reduce``. Any positive split count is valid; splits
    own balanced contiguous page ranges. Buffers are caller-owned.
    ``pipelined=False`` is the fully waited diagnostic schedule.
    """
    _require(str(get_rocm_arch()).split(":", 1)[0] == "gfx1250", "expected gfx1250")
    _require(q.is_cuda and q.shape[1:] == (64, 576), "expected GPU native H64 Q")
    _require_layout("q", q, torch.float8_e4m3fn, (None, 64, 576))
    _require_layout("kv", kv, torch.float8_e4m3fn, (None, 64, 1, 576))
    batch = q.shape[0]
    _require(isinstance(num_splits, int) and num_splits > 0, "invalid splits")
    _require(kv_num_stages in (3, 4), "invalid stage count")
    _require(pv_load_depth in (1, 4, 8), "invalid PV load depth")
    _require(waves_per_eu in (0, 1, 2), "invalid occupancy hint")
    _require_layout("block_tables", block_tables, torch.int32, (batch, None))
    _require_layout("seqused_k", seqused_k, torch.int32, (batch,))
    _require_layout("q_scale", q_scale, torch.float32, (1,))
    _require_layout("kv_scale", kv_scale, torch.float32, (1,))
    _require_layout("split_lse", split_lse, torch.float32, (batch, num_splits, 64, 1))
    shape = (batch, 64, 512) if num_splits == 1 else (batch, num_splits, 64, 512)
    mode = 0 if num_splits == 1 else (2 if partial_bf16 else 1)
    _require_layout("split_data", split_data, torch.float32 if mode == 1 else torch.bfloat16, shape)
    _require(all(t.device == q.device for t in (split_data, split_lse, kv, block_tables, seqused_k, q_scale, kv_scale)), "all tensors must be on q.device")
    from flydsl.compiler.kernel_function import CompilationContext
    from .kernels.mla_gfx1250.mla_pagesize64_fp8_fp8_ring import launch_mla_pagesize64_fp8_fp8_ring
    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    # "N,N" pins one register budget, as Triton's AMD backend does.
    with CompilationContext.compile_hints({"waves_per_eu": (waves_per_eu, waves_per_eu) if waves_per_eu else 0}):
        _run_compiled(
            launch_mla_pagesize64_fp8_fp8_ring,
            ptr_arg(split_data, fx.Float32 if mode == 1 else fx.BFloat16),
            ptr_arg(split_lse, fx.Float32), ptr_arg(q, fx.Int8), ptr_arg(kv, fx.Int8),
            ptr_arg(block_tables, fx.Int32), ptr_arg(seqused_k, fx.Int32),
            ptr_arg(q_scale, fx.Float32), ptr_arg(kv_scale, fx.Float32),
            float(softmax_scale), batch, num_splits, block_tables.shape[1],
            mode, kv_num_stages, int(bool(pipelined)), pv_load_depth, stream,
            specialization_key=(mode, kv_num_stages, waves_per_eu, int(bool(pipelined)), pv_load_depth),
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
    # BF16 partials (opt-in, native ring) are read and widened to FP32.
    partial_bf16 = split_data.dtype == torch.bfloat16
    hints = {"partial_dtype": "bf16"} if partial_bf16 else {}
    launch = compile_mla_decode_reduce(H=num_heads, Dv=v_head_dim, out_dtype=out_dtype, **hints)
    if stream is None:
        stream = torch.cuda.current_stream(out.device)
    out_type = fx.BFloat16 if out_dtype == "bf16" else fx.Float16
    _run_compiled(
        launch,
        ptr_arg(split_data, fx.BFloat16 if partial_bf16 else fx.Float32),
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


class MLADecodePagesize64Workspace:
    """Route and preallocated buffers for :func:`flydsl_mla_decode_pagesize64_fp8`.

    Make one with :func:`flydsl_mla_decode_pagesize64_fp8_workspace` and reuse
    it for every call (and graph replay) with the same heads, batch,
    ``max_seqlen_k`` and block-table width.
    """

    def __init__(self, key, route, bucket, schedule, num_splits, buffers):
        self.key = key
        self.route = route  # "native" (measured ring) or "fallback" (packed old kernel)
        self.bucket = bucket
        self.schedule = schedule
        self.num_splits = num_splits
        self.buffers = buffers


def _mla_decode_route(heads, batch, max_seqlen_k):
    """Host-only route lookup; never reads device memory."""
    if heads != 64 or max_seqlen_k <= 0:
        return "fallback", "native-ring supports H64 with a known max_seqlen_k", None
    from .mla_ring_config import select_mla_ring_config

    bucket, schedule = select_mla_ring_config(heads, batch, max_seqlen_k)
    return ("native" if schedule else "fallback"), bucket, schedule


def _mla_fallback_splits(batch, max_pages):
    """Coverage heuristic, not tuned: the smallest power of two with
    batch * splits >= CU count, capped at 64 and at the page count."""
    from aiter.jit.utils.chip_info import get_cu_num

    splits = 1
    while batch * splits < int(get_cu_num()) and splits < 64:
        splits *= 2
    return max(1, min(splits, max_pages))


def flydsl_mla_decode_pagesize64_fp8_workspace(q, kv, block_tables, max_seqlen_k):
    """Select the route and allocate all buffers for one decode shape.

    Args: as :func:`flydsl_mla_decode_pagesize64_fp8`. Only shapes, the device
    and the host integer ``max_seqlen_k`` are used; no device data is read.

    Returns: a :class:`MLADecodePagesize64Workspace`. The fallback route also
    holds a packed copy of the referenced pages, ``batch * width`` pages of
    36864 bytes plus a gather buffer of the same size, where
    ``width = min(block_tables.shape[1], ceil(max_seqlen_k / 64))``
    (the full table width when ``max_seqlen_k`` is 0).
    """
    batch, heads = q.shape[:2]
    device = q.device
    max_seqlen_k = int(max_seqlen_k)
    route, bucket, schedule = _mla_decode_route(heads, batch, max_seqlen_k)
    width = block_tables.shape[1]
    if max_seqlen_k > 0:
        width = min(width, (max_seqlen_k + 63) // 64)
    key = (heads, batch, max_seqlen_k, block_tables.shape[1], device)
    int32 = dict(device=device, dtype=torch.int32)
    if route == "native":
        splits = schedule["num_splits"]
        buffers = {}
    else:
        splits = _mla_fallback_splits(batch, width)
        pages = batch * width
        buffers = {
            "gather": torch.empty((pages, 64 * 576), device=device, dtype=torch.uint8),
            "packed": torch.empty((pages, 64 * 576), device=device, dtype=torch.float8_e4m3fn),
            "pages": torch.empty((batch,), **int32),
            "kv_indptr": torch.zeros((batch + 1,), **int32),
            "page_indices": torch.zeros((pages + 1,), **int32),
            "last_page_lens": torch.empty((batch,), **int32),
            "qo_indptr": torch.arange(batch + 1, **int32),
            "split_indptr": torch.arange(batch + 1, **int32) * splits,
            "column": torch.arange(width, **int32).view(1, width),
            "source": torch.arange(pages, **int32),
            "valid": torch.empty((batch, width), device=device, dtype=torch.bool),
            "safe": torch.empty((batch, width), **int32),
            "dest": torch.empty((batch, width), device=device, dtype=torch.int64),
            "spare": torch.full((1, 1), pages, device=device, dtype=torch.int64),
        }
    buffers["lse"] = torch.empty((batch, splits, heads, 1), device=device, dtype=torch.float32)
    if splits > 1:
        bf16 = route == "native" and schedule.get("partial_bf16", False)
        buffers["partial"] = torch.empty((batch, splits, heads, 512), device=device,
                                         dtype=torch.bfloat16 if bf16 else torch.float32)
    return MLADecodePagesize64Workspace(key, route, bucket, schedule, splits, buffers)


def _pack_referenced_pages(kv, block_tables, seqused_k, ws):
    """Device-only metadata and packing for the old kernel; no host sync."""
    b = ws.buffers
    batch, width = b["valid"].shape
    pages = batch * width
    torch.add(seqused_k, 63, out=b["pages"])
    torch.floor_divide(b["pages"], 64, out=b["pages"])
    torch.cumsum(b["pages"], 0, dtype=torch.int32, out=b["kv_indptr"][1:])
    torch.sub(seqused_k, 1, out=b["last_page_lens"])
    torch.remainder(b["last_page_lens"], 64, out=b["last_page_lens"])
    b["last_page_lens"].add_(1)
    # Unused table columns read column 0, a valid page of a nonempty sequence.
    torch.lt(b["column"], b["pages"].view(batch, 1), out=b["valid"])
    torch.where(b["valid"], block_tables[:, :width], block_tables[:, :1], out=b["safe"])
    # Compact CSR: entry kv_indptr[s] + j holds packed page s * width + j.
    # Unused columns write to the spare last slot.
    torch.add(b["kv_indptr"][:-1].view(batch, 1), b["column"], out=b["dest"])
    torch.where(b["valid"], b["dest"], b["spare"], out=b["dest"])
    b["page_indices"].scatter_(0, b["dest"].view(-1), b["source"])
    # Native token rows [64, 576] -> old page [64 x 512 NoPE | 64 x 64 RoPE].
    torch.index_select(kv.view(torch.uint8).view(-1, 64 * 576), 0, b["safe"].view(-1), out=b["gather"])
    rows = b["gather"].view(pages, 64, 576)
    packed = b["packed"].view(torch.uint8)
    packed[:, : 64 * 512].view(pages, 64, 512).copy_(rows[:, :, :512])
    packed[:, 64 * 512 :].view(pages, 64, 64).copy_(rows[:, :, 512:])


def flydsl_mla_decode_pagesize64_fp8(
    q, kv, block_tables, seqused_k, max_seqlen_k, q_scale, kv_scale,
    softmax_scale, out, *, workspace=None, stream=None,
):
    """Decode one FP8 query per sequence from native page64 FP8 KV on gfx1250.

    Computes ``out = softmax(q @ K^T * softmax_scale * q_scale * kv_scale) @ V
    * kv_scale`` per sequence and head, with K = all 576 columns and V = the
    first 512 columns of each KV token row.

    Route: a measured bucket of the native-ring table (H64 only, keyed by H,
    batch and ``max_seqlen_k``) runs the native ring kernel with that bucket's
    schedule. Every other shape uses the old packed kernel as coverage, not for
    speed: the call packs the referenced pages and builds CSR metadata on the
    device inside the call.

    Args:
        q: FP8 [batch, 8/64/128, 576], head stride 576, contiguous.
        kv: FP8 [pages, 64, 1, 576] token-interleaved pages, contiguous.
        block_tables: int32 [batch, width] physical page per logical page.
        seqused_k: int32 [batch] KV length per sequence.
        max_seqlen_k: host int, an upper bound of ``seqused_k`` (0 = unknown,
            which selects the fallback and full table width).
        q_scale, kv_scale: FP32 [1] dequantization scales.
        softmax_scale: host float.
        out: BF16 [batch, heads, 512], contiguous; overwritten.
        workspace: from :func:`flydsl_mla_decode_pagesize64_fp8_workspace`;
            allocated here when None (not graph-capturable then).
        stream: optional stream; default is the current stream.

    Returns: ``out``.

    Limits: every sequence is nonempty, ``seqused_k <= max_seqlen_k`` when
    nonzero, and the first ``ceil(seqused_k / 64)`` table entries of each
    sequence are valid pages of ``kv``. These are not checked on the device and
    no metadata is copied to the host, so a call with a workspace is graph-safe.
    """
    _require(isinstance(max_seqlen_k, int) and max_seqlen_k >= 0, "max_seqlen_k must be a host int >= 0")
    _require(q.ndim == 3 and q.shape[1] in (8, 64, 128), "q must be [batch, 8/64/128, 576]")
    _require_layout("q", q, torch.float8_e4m3fn, (None, q.shape[1], 576))
    batch, heads = q.shape[:2]
    _require_layout("kv", kv, torch.float8_e4m3fn, (None, 64, 1, 576))
    _require_layout("block_tables", block_tables, torch.int32, (batch, None))
    _require_layout("seqused_k", seqused_k, torch.int32, (batch,))
    _require_layout("out", out, torch.bfloat16, (batch, heads, 512))
    if workspace is None:
        workspace = flydsl_mla_decode_pagesize64_fp8_workspace(q, kv, block_tables, max_seqlen_k)
    _require(workspace.key == (heads, batch, max_seqlen_k, block_tables.shape[1], q.device),
             "workspace was made for another shape")
    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    b = workspace.buffers
    splits = workspace.num_splits
    partial = out if splits == 1 else b["partial"]
    with torch.cuda.stream(stream):
        if workspace.route == "native":
            schedule = dict(workspace.schedule)
            schedule.pop("num_splits")
            flydsl_mla_pagesize64_fp8_fp8_ring(
                partial, b["lse"], q, kv, block_tables, seqused_k, q_scale, kv_scale,
                softmax_scale, splits, stream=stream, **schedule,
            )
        else:
            _pack_referenced_pages(kv, block_tables, seqused_k, workspace)
            flydsl_mla_pagesize64_fp8_fp8(
                partial, b["lse"], q, b["packed"], b["kv_indptr"], b["page_indices"],
                b["last_page_lens"], b["qo_indptr"], b["split_indptr"], q_scale, kv_scale,
                softmax_scale, splits, stream=stream,
            )
        if splits > 1:
            flydsl_mla_decode_reduce(partial, b["lse"], seqused_k, out, splits, 1, stream=stream)
    return out
