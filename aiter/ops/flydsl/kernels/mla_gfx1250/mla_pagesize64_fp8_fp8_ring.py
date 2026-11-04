# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native FP8 page64 transfer and WMMA building blocks for gfx1250 MLA."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, tdm_ops
from flydsl.expr import arith as farith
from flydsl.expr import math as fmath
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr.typing import T, Vector as Vec

from ..gemm_common_gfx1250 import make_lds_copy_ops
from ..tensor_shim import ptr_buf_tensor
from .mla_common import (
    _concat_ds_tr8_b64, _concat_wmma_operand_k64, _rmem_i32, _xor16_f32,
    make_fp8_wmma_atom,
)

Q_BYTES = 64 * (528 + 80)
SLOT_BYTES = Q_BYTES
# QK runs with 16 heads per wave; PV runs with 128 output columns per wave over
# all 64 heads. Each wave publishes its packed FP8 probabilities (one PV B
# operand, 32 bytes per lane) and per-head alpha through this exchange area.
P_EXCHANGE_BYTES = 4 * 32 * 32
ALPHA_BYTES = 64 * 4
EXCHANGE_BYTES = P_EXCHANGE_BYTES + ALPHA_BYTES


def plain_barrier():
    """Workgroup barrier without a memory fence; callers add the waits they need."""
    rocdl_dialect.s_barrier_signal(-1)
    rocdl_dialect.s_barrier_wait(-1)


@flyc.jit
def native_page_load(global_ptr: fx.Pointer, shared_ptr: fx.Pointer,
                     columns: fx.Constexpr[int], pitch: fx.Constexpr[int]):
    """One wave copies 16 native token rows to a padded LDS tile."""
    global_tile = fx.Tensor(fx.make_view(
        global_ptr, fx.make_layout((16, columns), (576, 1))))
    shared_tile = fx.Tensor(fx.make_view(
        shared_ptr, fx.make_layout(((16, columns), 1), ((pitch, 1), 16 * pitch))))
    atom, coords = rocdl.cdna5.make_tiled_tdm_atom(
        rocdl.TensorLoad(), global_tile,
        fx.make_layout((16, columns), (pitch, 1)), (16, columns), num_warps=1)
    tile = fx.zipped_divide(coords, (16, columns))[None, (0, None)]
    shared, global_ = rocdl.cdna5.tdm_partition(
        atom, 0, fx.make_layout(1, 1), shared_tile, tile)
    fx.copy(atom, global_[None, 0], shared[None, 0])


@flyc.jit
def row_tile_store(global_ptr: fx.Pointer, shared_ptr: fx.Pointer,
                   rows: fx.Constexpr[int], columns: fx.Constexpr[int],
                   global_stride: fx.Constexpr[int]):
    """One wave stores a packed LDS tile; TDM stores cannot remove padding."""
    global_tile = fx.Tensor(fx.make_view(
        global_ptr, fx.make_layout((rows, columns), (global_stride, 1))))
    shared_tile = fx.Tensor(fx.make_view(
        shared_ptr, fx.make_layout(((rows, columns), 1), ((columns, 1), rows * columns))))
    atom, coords = rocdl.cdna5.make_tiled_tdm_atom(
        rocdl.TensorStore(), global_tile,
        fx.make_layout((rows, columns), (columns, 1)), (rows, columns), num_warps=1)
    tile = fx.zipped_divide(coords, (rows, columns))[None, (0, None)]
    shared, global_ = rocdl.cdna5.tdm_partition(
        atom, 0, fx.make_layout(1, 1), shared_tile, tile)
    fx.copy(atom, shared[None, 0], global_[None, 0])


@flyc.jit
def launch_mla_pagesize64_fp8_fp8_ring(
    output: fx.Pointer, lse: fx.Pointer, q: fx.Pointer, kv: fx.Pointer,
    page_table: fx.Pointer, lengths: fx.Pointer,
    q_scale: fx.Pointer, kv_scale: fx.Pointer, softmax_scale: fx.Float32,
    batch: fx.Int32, splits: fx.Int32, table_stride: fx.Int32,
    split_output: fx.Constexpr[int], stages: fx.Constexpr[int] = 3,
    pipelined: fx.Constexpr[int] = 0,
    pv_load_depth: fx.Constexpr[int] = 1,
    stream: fx.Stream = fx.Stream(None),
):
    """H64 native MLA; diagnostic or separately waited ordinary ring.

    split_output: 0 = BF16 final output (one split), 1 = normalized FP32
    partials, 2 = normalized BF16 partials. LSE is always FP32.
    """
    assert stages in (3, 4)
    assert pv_load_depth in (1, 4, 8)
    prefetch = stages - 1
    exchange_base = (stages + 1) * SLOT_BYTES
    alpha_base = exchange_base + P_EXCHANGE_BYTES
    lds_bytes = exchange_base + EXCHANGE_BYTES

    @flyc.kernel(known_block_size=[128, 1, 1])
    def attention(output: fx.Pointer, lse: fx.Pointer, q: fx.Pointer,
                  kv: fx.Pointer, page_table: fx.Pointer, lengths: fx.Pointer,
                  q_scale: fx.Pointer, kv_scale: fx.Pointer,
                  softmax_scale: fx.Float32, splits: fx.Int32,
                  table_stride: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        wave = rocdl.readfirstlane(T.i32, tid >> 5)
        lane = tid & 31
        row = lane & 15
        half = lane >> 4
        seq = fx.Int32(fx.block_idx.x)
        segment = fx.Int32(fx.block_idx.y)
        head = wave * 16 + row
        seq_len = ptr_buf_tensor(lengths, fx.Int32)[seq]
        num_pages = (seq_len + 63) // 64
        # Balanced contiguous segments (Gluon-style page locality). Segment s
        # owns pages [first_page, first_page + pages); the nonempty segments
        # are the dense prefix min(num_pages, splits) the reducer expects.
        base_pages = num_pages // splits
        extra_pages = num_pages - base_pages * splits
        pages = base_pages + (segment < extra_pages).select(fx.Int32(1), fx.Int32(0))
        first_page = segment * base_pages + (segment < extra_pages).select(segment, extra_pages)
        shared = fx.SharedAllocator(static=False).allocate(lds_bytes)._ptr
        shared_idx = fx.index_cast(T.index, fx.ptrtoint(shared))
        load128, store128 = make_lds_copy_ops(128)
        load32, store32 = make_lds_copy_ops(32)
        q_base = stages * SLOT_BYTES
        q_nope = q_base + head * 528
        q_rope = q_base + 64 * 528 + head * 80
        q_global = fx.add_offset(q, fx.Int64(seq) * 64 * 576 + fx.Int64(wave) * 16 * 576)
        native_page_load(q_global, fx.add_offset(shared, q_base + wave * 16 * 528), 512, 528)
        native_page_load(fx.add_offset(q_global, 512), fx.add_offset(shared, q_base + 64 * 528 + wave * 16 * 80), 64, 80)
        tdm_ops.tensor_wait(0)
        gpu.barrier()
        scale = softmax_scale * ptr_buf_tensor(q_scale, fx.Float32)[0] * ptr_buf_tensor(kv_scale, fx.Float32)[0] * 1.4426950408889634
        value_scale = ptr_buf_tensor(kv_scale, fx.Float32)[0]
        wmma64 = make_fp8_wmma_atom(64)
        neg_inf = fx.Float32(float('-inf'))
        # PV layout: this wave owns output columns [wave * 128, wave * 128 + 128).
        v_row = (lane >> 3) * 4 + (lane & 3)
        v_col = ((lane & 7) >> 2) * 8 + wave * 128
        p_publish = exchange_base + wave * 1024 + lane * 32
        p_gather = exchange_base + lane * 32

        def issue(page_number, slot):
            page_number = fx.Int32(page_number)
            slot = fx.Int32(slot)
            physical = ptr_buf_tensor(page_table, fx.Int32)[seq * table_stride + page_number]
            native = fx.add_offset(kv, fx.Int64(physical) * 64 * 576 + fx.Int64(wave) * 16 * 576)
            native_page_load(native, fx.add_offset(shared, slot * SLOT_BYTES + wave * 16 * 528), 512, 528)
            native_page_load(fx.add_offset(native, 512), fx.add_offset(shared, slot * SLOT_BYTES + 64 * 528 + wave * 16 * 80), 64, 80)

        def qk_part(slot, scores, rope=False):
            # Each K64 Q fragment is read once and reused by the four
            # independent token-tile accumulators (round-robin). LDS waits
            # come from the compiler's per-operand dependency tracking.
            accs = []
            for tile in range_constexpr(4):
                acc = fx.make_rmem_tensor(8, fx.Float32)
                acc.store(scores[tile])
                accs.append(acc)
            if const_expr(rope):
                k_base = slot * SLOT_BYTES + 64 * 528 + row * 80 + half * 16
                q_frags, k_pitch, k_tile_bytes = [q_rope + half * 16], 80, 16 * 80
            else:
                k_base = slot * SLOT_BYTES + row * 528 + half * 16
                q_frags, k_pitch, k_tile_bytes = [q_nope + half * 16 + f * 64 for f in range_constexpr(8)], 528, 16 * 528
            for f, q_addr in enumerate(q_frags):
                q_operand = _rmem_i32(8, _concat_wmma_operand_k64([Vec(load128(shared_idx, q_addr + c * 32)) for c in range_constexpr(2)]))
                k_operands = [
                    _rmem_i32(8, _concat_wmma_operand_k64([Vec(load128(shared_idx, k_base + tile * k_tile_bytes + f * 64 + c * 32)) for c in range_constexpr(2)]))
                    for tile in range_constexpr(4)
                ]
                for tile in range_constexpr(4):
                    fx.gemm(wmma64, accs[tile], k_operands[tile], q_operand, accs[tile])
            return [Vec(acc.load()) for acc in accs]

        def consume(tile_index, slot, state, fill=False, lora_wait=0, rope_wait=0, masked=True):
            tile_index = fx.Int32(tile_index)
            slot = fx.Int32(slot)
            # LoRA visible to all waves, and every wave's earlier LDS reads are
            # complete, so the refill below may overwrite the previous slot.
            tdm_ops.tensor_wait(lora_wait)
            rocdl.s_wait_dscnt(0)
            plain_barrier()
            scores = qk_part(slot, [fx.full(8, 0.0, fx.Float32) for _ in range_constexpr(4)])
            if const_expr(fill):
                # Every wave finished the previous iteration's PV reads before
                # the barrier above, so the refill slot is free.
                future = tile_index + prefetch
                issue(first_page + future, (slot + prefetch) % stages)
            # RoPE visibility only: no LDS location is rewritten here, so the
            # LoRA K reads in flight need no drain.
            tdm_ops.tensor_wait(rope_wait)
            plain_barrier()
            scores = qk_part(slot, scores, rope=True)
            old_max = fx.Float32(state[0])
            old_sum = fx.Float32(state[1])
            page_number = first_page + tile_index
            # Full pages skip all bounds work; only the masked (tail-capable)
            # path compares tokens.
            logits = []
            for tile in range_constexpr(4):
                logit = Vec(scores[tile]) * scale
                if const_expr(masked):
                    logit = Vec.from_elements([
                        (page_number * 64 + tile * 16 + half * 8 + i < seq_len).select(logit[i], neg_inf)
                        for i in range_constexpr(8)], fx.Float32)
                logits.append(logit)
            # Balanced packed reductions: tiles, then 8 -> 4 -> 2 -> 1 lanes of
            # the vector, then the paired half-wave.
            def tree(values, combine):
                v = combine(combine(values[0], values[1]), combine(values[2], values[3]))
                v = combine(v.shuffle(v, [0, 1, 2, 3]), v.shuffle(v, [4, 5, 6, 7]))
                v = combine(v.shuffle(v, [0, 1]), v.shuffle(v, [2, 3]))
                return combine(v[0], v[1])

            def vmax(x, y):
                return Vec(farith.maxnumf(x, y)) if isinstance(x, Vec) else fx.Float32(farith.maxnumf(x, y))

            maximum = fx.Float32(farith.maxnumf(tree(logits, vmax), old_max))
            maximum = fx.Float32(farith.maxnumf(maximum, _xor16_f32(maximum)))
            alpha = fx.Float32(rocdl.exp2(T.f32, (old_max - maximum).ir_value()))
            probs = []
            for tile in range_constexpr(4):
                shifted = logits[tile] - maximum
                probs.append(Vec.from_elements([rocdl.exp2(T.f32, shifted[i].ir_value()) for i in range_constexpr(8)], fx.Float32))
            row_sum = tree(probs, lambda x, y: x + y)
            denominator = old_sum * alpha + (row_sum + _xor16_f32(row_sum))
            one = fx.Float32(1.0).ir_value()
            pk8 = [Vec(rocdl_dialect.cvt_scalef32_pk8_fp8_f32(T.vec(2, T.i32), probs[tile], one)) for tile in range_constexpr(4)]
            packed = pk8[0].shuffle(pk8[1], [0, 1, 2, 3]).shuffle(pk8[2].shuffle(pk8[3], [0, 1, 2, 3]), list(range(8)))
            if const_expr(masked):
                # Zero this wave's own V columns for invalid token rows: a zero
                # probability does not cancel NaN. Other waves only read these
                # bytes as K for tokens whose logits are already -inf, and this
                # wave's later V reads are ordered after its own LDS stores.
                valid_rows = seq_len - page_number * 64
                zeros = fx.full(4, 0, fx.Int32)
                for k in range_constexpr(16):
                    chunk = lane + 32 * k
                    if chunk // 8 >= valid_rows:
                        store128(shared_idx, slot * SLOT_BYTES + (chunk // 8) * 528 + wave * 128 + (chunk % 8) * 16, zeros)
            # Publish this wave's 16-head P operand and alpha, then gather all
            # 64 heads. The next write to this area follows the next
            # iteration's barriers, after every wave consumed these values.
            for c in range_constexpr(2):
                store128(shared_idx, p_publish + c * 16, Vec.from_elements([packed[c * 4 + i] for i in range_constexpr(4)], fx.Int32))
            store32(shared_idx, alpha_base + head * 4, Vec.from_elements([alpha], fx.Float32).bitcast(fx.Int32))
            gpu.barrier()
            p_operands = []
            alphas = []
            for m in range_constexpr(4):
                chunks = [Vec(load128(shared_idx, p_gather + m * 1024 + c * 16)) for c in range_constexpr(2)]
                p_operands.append(_rmem_i32(8, chunks[0].shuffle(chunks[1], list(range(8)))))
                alphas.append(fx.Float32(Vec(load32(shared_idx, alpha_base + (m * 16 + row) * 4)).bitcast(fx.Float32)[0]))
            def load_value(j):
                chunks = []
                for tile in range_constexpr(4):
                    ptr = fx.add_offset(shared, slot * SLOT_BYTES + (tile * 16 + v_row) * 528 + v_col + j * 16)
                    chunks.append(Vec(rocdl.ds_load_tr8_b64(T.vec(2, T.i32), ptr.llvm_ptr)))
                return _concat_ds_tr8_b64(chunks)

            staged_values = {}
            for j in range_constexpr(pv_load_depth):
                staged_values[j] = load_value(j)
            outs = [None] * 32
            for j in range_constexpr(8):
                if const_expr(j + pv_load_depth < 8):
                    staged_values[j + pv_load_depth] = load_value(j + pv_load_depth)
                v_operand = _rmem_i32(8, staged_values.pop(j))
                for m in range_constexpr(4):
                    acc = fx.make_rmem_tensor(8, fx.Float32)
                    acc.store(Vec(state[2 + m * 8 + j]) * alphas[m])
                    fx.gemm(wmma64, acc, v_operand, p_operands[m], acc)
                    outs[m * 8 + j] = Vec(acc.load())
            return [maximum, denominator] + outs

        if pages > 0:
            zero = fx.full(8, 0.0, fx.Float32)
            initial = [neg_inf, fx.Float32(0.0)] + [zero for _ in range_constexpr(32)]
            if const_expr(pipelined):
                for prime in range_constexpr(prefetch):
                    clamped = (fx.Int32(prime) < pages).select(fx.Int32(prime), pages - 1)
                    issue(first_page + clamped, fx.Int32(prime))
                steady = (pages > prefetch).select(pages - prefetch, fx.Int32(0))
                for tile, state in range(fx.Int32(0), steady, fx.Int32(1), init=initial):
                    tile = fx.Int32(tile)
                    steady_result = yield consume(tile, tile % stages, state, fill=True, lora_wait=2 * prefetch - 1, rope_wait=2 * prefetch, masked=False)
                # Drain (holds the only possibly partial page): runtime loop,
                # fully waited TDM, masked tail handling.
                for tile, state in range(steady, pages, fx.Int32(1), init=steady_result):
                    tile = fx.Int32(tile)
                    drain_result = yield consume(tile, tile % stages, state)
                final = drain_result
            else:
                for tile, state in range(fx.Int32(0), pages, fx.Int32(1), init=initial):
                    tile = fx.Int32(tile)
                    gpu.barrier()
                    issue(first_page + tile, fx.Int32(0))
                    result = yield consume(tile, fx.Int32(0), state)
                final = result
            # Short segments prime duplicate pages; drain their unused transfers
            # and every wave's last exchange reads before reusing LDS.
            tdm_ops.tensor_wait(0)
            gpu.barrier()
            inv = fx.Float32(rocdl.rcp(T.f32, fx.Float32(final[1]).ir_value()))
            store32(shared_idx, alpha_base + head * 4, Vec.from_elements([value_scale * inv], fx.Float32).bitcast(fx.Int32))
            gpu.barrier()
            output_scales = [fx.Float32(Vec(load32(shared_idx, alpha_base + (m * 16 + row) * 4)).bitcast(fx.Float32)[0]) for m in range_constexpr(4)]
            # Reclaim the dead ring as one packed [64 heads, 128 columns] tile
            # per wave; TDM cannot remove padding.
            element_bytes = 4 if split_output == 1 else 2
            tile_base = wave * 64 * 128 * element_bytes
            for m in range_constexpr(4):
                for j in range_constexpr(8):
                    values = Vec(final[2 + m * 8 + j]) * output_scales[m]
                    offset = tile_base + ((m * 16 + row) * 128 + j * 16 + half * 8) * element_bytes
                    if const_expr(split_output == 1):
                        for chunk in range_constexpr(2):
                            store128(shared_idx, offset + chunk * 16, Vec.from_elements([values[chunk * 4 + i] for i in range_constexpr(4)], fx.Float32).bitcast(fx.Int32))
                    else:
                        store128(shared_idx, offset, values.to(fx.BFloat16).bitcast(fx.Int32))
            rocdl.s_wait_dscnt(0)
            gpu.barrier()
            output_type = fx.Float32 if split_output == 1 else fx.BFloat16
            typed_output = fx.recast_iter(fx.PointerType.get(output_type.ir_type, output.memspace, 16), output)
            typed_shared = fx.recast_iter(fx.PointerType.get(output_type.ir_type, shared.memspace, 16), shared)
            base = (fx.Int64(seq) * splits + segment) * 64 * 512 if split_output else fx.Int64(seq) * 64 * 512
            row_tile_store(fx.add_offset(typed_output, base + wave * 128), fx.add_offset(typed_shared, wave * 64 * 128), 64, 128, 512)
            tdm_ops.tensor_wait(0)
            if half == 0:
                ln_sum = fmath.log(fx.Float32(final[1]))
                ptr_buf_tensor(lse, fx.Float32)[(seq * splits + segment) * 64 + head] = (fx.Float32(final[0]) * 0.6931471805599453 + ln_sum)

    attention(output, lse, q, kv, page_table, lengths, q_scale, kv_scale,
              softmax_scale, splits, table_stride).launch(
        grid=(batch, splits), block=(128,), stream=stream)


# Match the Triton AMD backend's gfx1250 codegen for the Gluon reference:
# co-execution scheduler plus expert scheduling mode.
launch_mla_pagesize64_fp8_fp8_ring.compile_hints = {
    "llvm_options": {
        "amdgpu-sched-strategy": "coexec",
        "amdgpu-expert-scheduling-mode": True,
    },
}
