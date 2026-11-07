# SPDX-License-Identifier: Apache-2.0
"""Switchable event recorder for UCLinker flow and timing diagnostics.

Enable with ``SGLANG_UCL_TRACE=1`` and ``SGLANG_UCL_TRACE_DIR=<run dir>``.
Each process appends JSONL records to ``<dir>/<host>_<pid>.jsonl`` from a
background thread; ``examples/runtime/engine/Round/ucl_trace_tool.py`` merges
the files into a Chrome/Perfetto timeline and a Markdown report.

Call sites must guard with ``if ucl_trace.ON:`` so a disabled recorder costs
one module attribute check and never builds arguments. Recording never
synchronizes CUDA, takes no I/O on the caller thread and does not change
control flow; if the directory cannot be written the recorder turns itself
off. Timestamps are ``time.perf_counter_ns()``; on Linux this is
CLOCK_MONOTONIC and therefore comparable across processes on one host only.

``SGLANG_UCL_TRACE_GPU=1`` adds timing events on the GPU streams, the one
exception to "never touches CUDA": they show when work really ran on the
device, and they sit in the stream, so keep it off for overhead comparisons.

This module is stdlib-only so the benchmark client can load it by path; torch
is imported only by the GPU functions.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import sys
import threading
import time
from collections import deque

_LEVEL = os.environ.get("SGLANG_UCL_TRACE", "0").strip()
_DIR = os.environ.get("SGLANG_UCL_TRACE_DIR", "").strip()
ON = _LEVEL not in ("", "0") and bool(_DIR)
GPU = ON and os.environ.get("SGLANG_UCL_TRACE_GPU", "0").strip() not in ("", "0")

FORMAT_VERSION = 1
FLUSH_INTERVAL_S = float(os.environ.get("SGLANG_UCL_TRACE_FLUSH_S", "1.0"))
MAX_BUFFERED = int(os.environ.get("SGLANG_UCL_TRACE_MAX_BUFFERED", "1000000"))
SLOW_NS = int(float(os.environ.get("SGLANG_UCL_TRACE_SLOW_MS", "2")) * 1e6)

now = time.perf_counter_ns

_events: deque = deque()
_stats: dict = {}
_stats_lock = threading.Lock()
_dropped = 0
_writer = None
_file = None
_identity = None
_closed = True  # until _start()
_init_lock = threading.Lock()
_gpu_anchor = None


def emit(name, rid=None, **fields):
    """Record an instant event."""
    _push((now(), -1, name, rid, threading.get_ident(), fields or None))


def span(name, start_ns, rid=None, **fields):
    """Record a completed interval that started at ``start_ns``."""
    end = now()
    _push((start_ns, end - start_ns, name, rid, threading.get_ident(), fields or None))


def stat(name, start_ns, slow_ns=SLOW_NS, **fields):
    """Aggregate a frequent call; keep an individual span only when slow."""
    end = now()
    duration = end - start_ns
    with _stats_lock:
        entry = _stats.get(name)
        if entry is None:
            _stats[name] = [1, duration, duration]
        else:
            entry[0] += 1
            entry[1] += duration
            if duration > entry[2]:
                entry[2] = duration
    if slow_ns is not None and duration >= slow_ns:
        _push((start_ns, duration, name, None, threading.get_ident(), fields or None))


def gpu_mark(stream):
    """A timing event that fires when ``stream`` reaches this point."""
    import torch

    event = torch.cuda.Event(enable_timing=True)
    event.record(stream)
    return event


def gpu_span(name, start, end, **fields):
    """Record the device interval between two finished marks on this clock.
    One anchor per process maps GPU event time onto ``now()``."""
    global _gpu_anchor
    import torch

    try:
        with _init_lock:
            if _gpu_anchor is None:
                anchor = gpu_mark(torch.cuda.Stream())
                anchor.synchronize()
                _gpu_anchor = (anchor, now())
        anchor, anchor_ns = _gpu_anchor
        start_ns = anchor_ns + int(anchor.elapsed_time(start) * 1e6)
        duration = int(start.elapsed_time(end) * 1e6)
    except Exception as error:  # a diagnostic must not fail a forward or GET
        emit("gpu.error", name=name, error=repr(error)[:200])
        return
    _push((start_ns, duration, name, None, threading.get_ident(), fields or None))


def gpu_streams(owner, **streams):
    """Record which CUDA streams this process uses, so aliased ones show up."""
    emit(
        "gpu.streams",
        owner=owner,
        max_connections=os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
        streams={
            name: [stream.stream_id, hex(stream.cuda_stream)]
            for name, stream in streams.items()
            if stream is not None
        },
    )


def set_identity(role, label=None, **fields):
    """Name this process (e.g. role=prefill, label=scheduler_DP0_TP0)."""
    global _identity
    if label is None:
        label = _process_title()
    _identity = dict(role=str(role), label=label, **fields)
    _push((now(), -1, "process.identity", None, threading.get_ident(), _identity))


def _push(record):
    # Callers only append. The file is created and written by the writer
    # thread, which turns the recorder off if the directory fails.
    global _dropped
    if _closed:
        return
    if len(_events) >= MAX_BUFFERED:
        _dropped += 1
        return
    _events.append(record)


def _process_title():
    try:
        import setproctitle

        return setproctitle.getproctitle()
    except Exception:
        return f"pid{os.getpid()}"


def _open():
    """Create the process file on first data, so idle importers leave none."""
    global _file
    os.makedirs(_DIR, exist_ok=True)
    host = socket.gethostname()
    path = os.path.join(_DIR, f"{host}_{os.getpid()}.jsonl")
    _file = open(path, "a", encoding="utf-8")
    _write_line(
        dict(
            type="header",
            version=FORMAT_VERSION,
            run=os.environ.get("SGLANG_UCL_TRACE_RUN")
            or os.path.basename(os.path.normpath(_DIR)),
            host=host,
            pid=os.getpid(),
            level=_LEVEL,
            clock="perf_counter_ns",
            perf_anchor_ns=now(),
            wall_anchor_ns=time.time_ns(),
            flush_interval_s=FLUSH_INTERVAL_S,
            slow_ms=SLOW_NS / 1e6,
        )
    )


def _write_line(obj):
    _file.write(json.dumps(obj, separators=(",", ":"), default=str))
    _file.write("\n")


def _drain():
    """Write buffered records and one flush marker; writer thread or atexit."""
    global _stats
    if _file is None:
        if not _events and not _stats:
            return
        _open()
    count = 0
    names = {t.ident: t.name for t in threading.enumerate()}
    while True:
        try:
            t, d, n, r, th, f = _events.popleft()
        except IndexError:
            break
        record = {"t": t, "n": n, "th": names.get(th, th)}
        if d >= 0:
            record["d"] = d
        if r is not None:
            record["r"] = r
        if f:
            record["f"] = f
        _write_line(record)
        count += 1
    with _stats_lock:
        stats, _stats = _stats, {}
    flushed = now()
    for name, (calls, total, peak) in stats.items():
        _write_line(dict(type="stat", t=flushed, n=name, c=calls, s=total, m=peak))
    _write_line(dict(type="flush", t=flushed, events=count, dropped=_dropped))
    _file.flush()


def _disable(error):
    """Stop recording after an I/O failure; serving continues untouched."""
    global ON, _closed
    ON = False
    _closed = True
    _events.clear()
    try:
        sys.stderr.write(f"UCL trace disabled: {error!r}\n")
    except Exception:
        pass


def _writer_loop():
    while not _closed:
        time.sleep(FLUSH_INTERVAL_S)
        with _init_lock:
            if _closed:
                return
            try:
                _drain()
            except Exception as error:
                # The report flags the missing end marker and last flush time.
                _disable(error)
                return


def _start():
    global _writer, _closed, _file, _dropped, _init_lock, _stats_lock
    # Fresh locks too: a fork may copy one held by the parent's writer.
    _init_lock, _stats_lock = threading.Lock(), threading.Lock()
    _file, _dropped = None, 0
    _events.clear()
    _stats.clear()
    _closed = False
    _writer = threading.Thread(
        target=_writer_loop, name="ucl-trace-writer", daemon=True
    )
    _writer.start()


def close():
    """Flush everything and write the end marker (atexit, or tests)."""
    global _closed, ON
    with _init_lock:
        if _closed:
            return
        _closed = True
        ON = False
        try:
            _drain()
            if _file is not None:
                _write_line(dict(type="end", t=now(), dropped=_dropped))
                _file.close()
        except Exception as error:
            _disable(error)


if ON:
    # Started once at import (process startup); call sites never do setup.
    _start()
    atexit.register(close)
    if hasattr(os, "register_at_fork"):
        # A forked child gets an empty buffer, its own file and writer thread.
        os.register_at_fork(after_in_child=lambda: ON and _start())
