# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Measured native-ring MLA schedules, keyed by H, B and max KV length buckets.

The table uses the bucket keys of the Gluon MLA decode table, for example
``H_LEQ_64.B_LEQ_32.KV_LEQ_32768``. The smallest matching bound per axis wins
(H, then B, then KV). An empty entry means "no measured schedule": the caller
uses its fallback. The ``any`` entry is required and is empty.
"""

import hashlib
import json
import logging
from pathlib import Path

from aiter.jit.utils.chip_info import get_cu_num, get_gfx

_CONFIG_ROOT = Path(__file__).parent / "configs" / "mla"
_CONTRACT = "native-h64-fp8-page64-q576-kv576-natural-lse-v2"
_SCHEMA = 2
_AXES = ("H", "B", "KV")
_CONFIG_KEYS = {"num_splits", "kv_num_stages", "waves_per_eu", "pipelined", "pv_load_depth"}
_TOP_KEYS = {"schema", "contract", "arch", "cu_num", "implementation_sha256", "buckets",
             "evidence", "selection_scope"}
_LOG = logging.getLogger(__name__)
_CACHE = {}
_WARNED = set()


def implementation_hash():
    """SHA-256 of the ring kernel source and the helper sources it imports."""
    kernel = Path(__file__).parent / "kernels" / "mla_gfx1250" / "mla_pagesize64_fp8_fp8_ring.py"
    digest = hashlib.sha256(kernel.read_bytes())
    for dependency in (kernel.parent / "mla_common.py", kernel.parent.parent / "gemm_common_gfx1250.py"):
        digest.update(dependency.read_bytes())
    return digest.hexdigest()


def _bound(part):
    axis, _, bound = part.partition("_LEQ_")
    if axis not in _AXES or not bound.isdigit():
        raise RuntimeError("Invalid native MLA ring bucket component: %s" % part)
    return axis, int(bound)


def _slots(key):
    """Expand a bucket key to one (axis, bound or None) slot per axis."""
    slot = dict.fromkeys(_AXES)
    if key != "any":
        for part in key.split("."):
            axis, bound = _bound(part)
            if slot[axis] is not None:
                raise RuntimeError("Repeated axis in native MLA ring bucket: %s" % key)
            slot[axis] = bound
    return tuple(slot[a] for a in _AXES)


def _validate_entry(key, entry):
    if not entry:
        return
    if (set(entry) != _CONFIG_KEYS
            or entry["num_splits"] not in (1, 2, 4, 8, 16, 32, 64)
            or entry["kv_num_stages"] not in (3, 4)
            or entry["waves_per_eu"] not in (1, 2)
            or entry["pv_load_depth"] not in (1, 4, 8)
            or entry["pipelined"] is not True):
        raise RuntimeError("Invalid native MLA ring schedule for bucket %s" % key)


def _table_path():
    arch = str(get_gfx()).split(":", 1)[0].removesuffix("-strict")
    cu = int(get_cu_num())
    return arch, cu, _CONFIG_ROOT / arch / ("%dcu" % cu) / "native_ring.json"


def _load_table():
    """Return the validated bucket table. Raise RuntimeError if missing or stale."""
    arch, cu, path = _table_path()
    if not path.exists():
        raise RuntimeError("No measured native MLA ring configuration: %s" % path)
    stamp = (str(path), path.stat().st_mtime_ns)
    if stamp in _CACHE:
        return _CACHE[stamp]
    table = json.loads(path.read_text())
    if (set(table) != _TOP_KEYS or table["schema"] != _SCHEMA
            or table["contract"] != _CONTRACT or table["arch"] != arch
            or table["cu_num"] != cu):
        raise RuntimeError("Incompatible native MLA ring configuration: %s" % path)
    if table["implementation_sha256"] != implementation_hash():
        raise RuntimeError("Stale native MLA ring configuration (kernel changed): %s" % path)
    buckets = table["buckets"]
    if buckets.get("any") != {}:
        raise RuntimeError("Native MLA ring configuration needs an empty 'any' bucket")
    index = {}
    for key, entry in buckets.items():
        _validate_entry(key, entry)
        slots = _slots(key)
        if slots in index:
            raise RuntimeError("Duplicate native MLA ring bucket: %s" % key)
        index[slots] = key
    _CACHE.clear()
    _CACHE[stamp] = (buckets, index)
    return _CACHE[stamp]


def _select(buckets, index, values):
    """Same order as config_utils.select_leq_config with axes=(H, B, KV)."""
    bounds = {a: sorted({s[i] for s in index if s[i] is not None}) for i, a in enumerate(_AXES)}
    candidates = [[b for b in bounds[a] if values[a] <= b] + [None] for a in _AXES]
    for h in candidates[0]:
        for b in candidates[1]:
            for kv in candidates[2]:
                if (h, b, kv) in index:
                    key = index[(h, b, kv)]
                    return key, dict(buckets[key])
    raise RuntimeError("Native MLA ring configuration has no matching bucket")


def select_mla_ring_config(heads, batch, max_seqlen_k):
    """Return ``(bucket_key, schedule)`` for a routed call, or ``(reason, None)``.

    A missing, stale or invalid table gives ``None`` and one logged warning per
    reason, not an exception. An empty bucket gives ``None`` (no measurement).
    """
    try:
        buckets, index = _load_table()
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        reason = str(error)
        if reason not in _WARNED:
            _WARNED.add(reason)
            _LOG.warning("%s; routed MLA decode uses the packed fallback", reason)
        return reason, None
    key, entry = _select(buckets, index, {"H": heads, "B": batch, "KV": max_seqlen_k})
    return key, (entry or None)


def load_mla_ring_config(batch, length, heads=64):
    """Return a verified schedule for an explicit native-ring call; refuse missing or stale tuning."""
    buckets, index = _load_table()
    key, entry = _select(buckets, index, {"H": heads, "B": batch, "KV": length})
    if not entry:
        raise RuntimeError("Native MLA ring bucket %s has no measured schedule" % key)
    return entry
