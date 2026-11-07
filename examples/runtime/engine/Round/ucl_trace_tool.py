#!/usr/bin/env python3
"""Merge and analyze UCLinker trace files written by ``ucl_trace.py``.

    python ucl_trace_tool.py report  RUN_DIR [--wait 3]   # report.md + trace.json + summary.json
    python ucl_trace_tool.py merge   RUN_DIR [-o trace.json]
    python ucl_trace_tool.py compare RUN_DIR_A RUN_DIR_B

trace.json opens in chrome://tracing or ui.perfetto.dev. Naming follows what
L1 CPU events can prove: "worker observed" completion is not device
completion, and Python spans are not GPU execution intervals.
Stdlib only.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from collections import defaultdict

STATE_ORDER = (
    "等待READY",
    "READY已到·未提交",
    "本rank就绪·等其他rank",
    "load进行中",
    "prefix完成·等传输/放行",
    "失败处理中",
)
HINT = "READY已满足·被inflight限制（候选优化点）"


# ----------------------------------------------------------------- loading


class TraceFile:
    def __init__(self, path):
        self.path = path
        self.header = {}
        self.identity = {}
        self.events = []
        self.stats = []
        self.last_flush = None
        self.dropped = 0
        self.ended = False
        self.bad_lines = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    self.bad_lines += 1  # torn last line of a killed process
                    continue
                kind = rec.get("type")
                if kind is None:
                    if rec["n"] == "process.identity":
                        self.identity = rec.get("f") or {}
                    self.events.append(rec)
                elif kind == "header":
                    self.header = rec
                elif kind == "stat":
                    self.stats.append(rec)
                elif kind == "flush":
                    self.last_flush = rec["t"]
                    self.dropped = rec.get("dropped", 0)
                elif kind == "end":
                    self.ended = True
                    self.last_flush = rec["t"]
                    self.dropped = rec.get("dropped", 0)
        self.host = self.header.get("host", "?")
        self.role = self.identity.get("role", "?")
        self.label = self.identity.get("label", os.path.basename(path))

    @property
    def name(self):
        return f"{self.role} {self.label}@{self.host}"


class Run:
    def __init__(self, directory):
        self.dir = directory
        paths = sorted(
            os.path.join(directory, p)
            for p in os.listdir(directory)
            if p.endswith(".jsonl")
        )
        if not paths:
            sys.exit(f"No *.jsonl trace files in {directory}")
        self.files = [TraceFile(p) for p in paths]
        # One perf clock per host: use one anchor per host so processes on the
        # same host keep their exact relative order (no per-process jitter).
        self.hosts = sorted({f.host for f in self.files})
        offsets = {}
        for f in self.files:
            h = f.header
            if f.host not in offsets and h:
                offsets[f.host] = h["wall_anchor_ns"] - h["perf_anchor_ns"]
        self.cross_host = len(self.hosts) > 1
        self.events = []
        for index, f in enumerate(self.files):
            offset = f.offset = offsets.get(f.host, 0)
            for e in f.events:
                e["src"] = index
                e["w"] = e["t"] + offset
                self.events.append(e)
        self.events.sort(key=lambda e: e["w"])
        self.t0 = self.events[0]["w"] if self.events else 0
        self._index_requests()
        self._index_get_generations()
        self.window = None  # (lo, hi) once restricted to the client experiment

    def _index_requests(self):
        self.room_rid = {}
        # P and D can hold different rids for one room; keep them per role.
        self.room_role_rids = defaultdict(lambda: defaultdict(list))
        self.rid_room = {}
        for e in self.events:
            room = (e.get("f") or {}).get("room")
            # Room 0 means no handoff (health checks), not a shared identity.
            if (
                room
                and e.get("r")
                and e["n"]
                in (
                    "req.arrive",
                    "pd.prealloc",
                    "linker.plan",
                )
            ):
                self.room_rid.setdefault(room, e["r"])
                rids = self.room_role_rids[room][self.files[e["src"]].role]
                if e["r"] not in rids:
                    rids.append(e["r"])
                self.rid_room.setdefault(e["r"], room)

    def _index_get_generations(self):
        """GET batch numbers restart after a linker reset (flush). Tag every
        batch event with the reset generation so batches pair within one
        generation only. ``od.reset`` marks it; older traces fall back to the
        enqueue number going backwards. reset() drains the queue first, so a
        generation's events all precede the next generation's first enqueue."""
        gen, last = defaultdict(int), {}
        for e in self.events:
            n, src = e["n"], e["src"]
            if n == "od.reset":
                gen[src] += 1
                last.pop(src, None)
            elif n == "od.get.enqueue":
                batch = f_(e, "batch")
                if src in last and batch is not None and batch <= last[src]:
                    gen[src] += 1
                last[src] = batch
            if n.startswith("od.get.") or n == "uc.load.consumed":
                e["gen"] = gen[src]

    def batch_key(self, e):
        return (e["src"], e.get("gen", 0), f_(e, "batch"))

    def windowed(self, lo, hi):
        """A view holding only events in [lo, hi] (files and indexes shared)."""
        view = copy.copy(self)
        view.events = [e for e in self.events if lo <= e["w"] <= hi]
        view.window = (lo, hi)
        return view

    def rid_of(self, e):
        r = e.get("r")
        if r:
            return r.split("@", 1)[0]
        room = (e.get("f") or {}).get("room")
        return self.room_rid.get(room)

    def role_rids(self, room, role):
        return self.room_role_rids[room][role] if room in self.room_role_rids else []

    def peer_rids(self, rid):
        """Every rid that shares this rid's room, itself included."""
        room = self.rid_room.get(rid)
        if room is None:
            return [rid]
        return [r for rids in self.room_role_rids[room].values() for r in rids]

    def ms(self, w):
        return (w - self.t0) / 1e6


# ----------------------------------------------------------------- helpers


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round(q / 100 * len(values) + 0.5)) - 1))
    return values[k]


def fmt(v, digits=1):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def dist(values):
    return dict(
        n=len(values),
        p50=pct(values, 50),
        p99=pct(values, 99),
        max=max(values) if values else None,
        sum=sum(values) if values else 0.0,
    )


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def f_(e, key, default=None):
    return (e.get("f") or {}).get(key, default)


def merge_range(ranges, start, end):
    """Union [start, end) into sorted disjoint ranges (adjacent ranges join)."""
    out = []
    for a, b in sorted(list(ranges) + [(start, end)]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def covered_until(ranges, start):
    for a, b in ranges:
        if a <= start < b:
            return b
    return start


# ----------------------------------------------------------- completeness


def completeness(run):
    rows, problems, open_files = [], [], []
    for f in run.files:
        status = "完整" if f.ended else "无结束标记"
        if f.dropped:
            status += f"，丢弃 {f.dropped}"
            problems.append(f"{f.name} 丢弃了 {f.dropped} 条事件：相关统计偏小。")
        if not f.ended:
            open_files.append(f.label)
        if f.bad_lines:
            status += f"，损坏行 {f.bad_lines}"
        last = None if f.last_flush is None else run.ms(f.last_flush + f.offset) / 1000
        rows.append(
            [
                f.role,
                f.label,
                f.host,
                f.header.get("pid", "?"),
                len(f.events),
                fmt(last, 1),
                status,
            ]
        )
    if open_files:
        problems.append(
            f"{len(open_files)} 个进程没有结束标记（服务仍在运行或被 kill，"
            f"最后一次刷盘之后的事件缺失）：{', '.join(open_files)}。"
        )
    pairs = [
        ("od.get.enqueue", "od.get.done", run.batch_key),
        ("uc.save.begin", "uc.save.consumed", lambda e: (e["src"], f_(e, "session"))),
        ("linker.load_submit", "linker.load_applied", lambda e: (e["src"], e.get("r"))),
        ("client.send", "client.end", lambda e: e.get("r")),
    ]
    unclosed = []
    for start, end, key in pairs:
        opened = {}
        for e in run.events:
            if e["n"] == start:
                opened[key(e)] = e
            elif e["n"] == end:
                opened.pop(key(e), None)
        if opened:
            unclosed.append([start, end, len(opened)])
    return rows, problems, unclosed


# ------------------------------------------------------------------ client


def client_section(run):
    reqs = {}
    for e in run.events:
        if e["n"].startswith("client."):
            reqs.setdefault(e["r"], []).append(e)
    server_rids = {e.get("r") for e in run.events if e["n"] == "req.arrive"}
    per_round = defaultdict(lambda: defaultdict(list))
    detail = {}
    server_round = {}  # server rid -> round, for joining server-side sections
    via_sse = 0
    for rid, evs in reqs.items():
        send = next((e for e in evs if e["n"] == "client.send"), None)
        if send is None:
            continue
        tag = f_(send, "tag", rid)
        rnd = tag.split("-", 1)[0]
        sse = [e for e in evs if e["n"] == "client.sse"]
        # A router may drop or rewrite the sent rid; the OpenAI stream id is
        # the server rid (meta_info["id"]), so it joins either way.
        server = rid if rid in server_rids else None
        if server is None:
            for e in sse:
                sid = f_(e, "id") or ""
                for cand in (sid, sid.removeprefix("chatcmpl-")):
                    if cand in server_rids:
                        server = cand
                        break
                if server is not None:
                    via_sse += 1
                    break
        if server is not None:
            server_round[server] = rnd
        content = [e for e in sse if f_(e, "content")]
        end = next((e for e in evs if e["n"] == "client.end"), None)
        bucket = per_round[rnd]
        bucket["n"].append(1)
        if end is not None and f_(end, "ok"):
            bucket["ok"].append(1)
        if sse:
            bucket["first_frame"].append((sse[0]["w"] - send["w"]) / 1e6)
        if content:
            first = (content[0]["w"] - send["w"]) / 1e6
            bucket["first_content"].append(first)
            gaps = [(b["w"] - a["w"]) / 1e6 for a, b in zip(content, content[1:])]
            bucket["gap"].extend(gaps)
            detail[rid] = dict(first_content_ms=first, max_gap_ms=max(gaps, default=0))
            if server is not None:
                detail[server] = detail[rid]
        if end is not None:
            bucket["e2e"].append((end["w"] - send["w"]) / 1e6)
        bucket["matched"].append(int(server is not None))
    rows, summary = [], {}
    for rnd in sorted(per_round, key=lambda r: int(r.lstrip("r") or 0)):
        b = per_round[rnd]
        ff, fc, gap, e2e = (
            dist(b[k]) for k in ("first_frame", "first_content", "gap", "e2e")
        )
        summary[rnd] = dict(first_frame=ff, first_content=fc, gap=gap, e2e=e2e)
        rows.append(
            [
                rnd,
                f"{len(b['ok'])}/{len(b['n'])}",
                fmt(ff["p50"]),
                fmt(fc["p50"]),
                fmt(fc["max"]),
                fmt(gap["p50"]),
                fmt(gap["p99"]),
                fmt(gap["max"]),
                fmt(e2e["p50"]),
                f"{sum(b['matched'])}/{len(b['matched'])}",
            ]
        )
    return rows, summary, detail, bool(reqs), server_round, via_sse


# --------------------------------------------------- decode request states


def decode_states(run):
    """Per-request mutually exclusive D-side state segments plus hints."""
    plans = {}
    for e in run.events:
        if e["n"] == "linker.plan":
            plans.setdefault(e["r"], e)  # first rank to plan = representative
    out = {}
    for rid, plan in plans.items():
        src, room = plan["src"], f_(plan, "room")
        evs = [
            e
            for e in run.events
            if e["src"] == src
            and e["w"] >= plan["w"]
            and (run.rid_of(e) == rid or (room is not None and f_(e, "room") == room))
        ]
        f = run.files[src]
        cutoff = None if f.last_flush is None else f.last_flush + f.offset
        cut = run.window is not None and cutoff is not None and run.window[1] < cutoff
        if cut:
            cutoff = run.window[1]  # a windowed run stops at the window end
        out[rid] = s = _walk_states(plan, evs, cutoff)
        s["window_cut"] = cut and s["outcome"] == "open"
    return out


def _walk_states(plan, evs, cutoff=None):
    """``cutoff``: the process's last confirmed flush; an unfinished request
    stays in its last state until then (a stall leaves no further events)."""
    page = f_(plan, "page", 1)
    window = f_(plan, "window_pages", 1)
    group = f_(plan, "group", 1)
    prefix_pages = f_(plan, "prefix_len", 0) // page
    loaded_len, loaded_layers = f_(plan, "loaded_len", 0), 0
    num_layers = 0
    for e in evs:
        layers = f_(e, "layers")
        if e["n"] in ("notify.recv", "linker.load_submit") and layers:
            num_layers = max(num_layers, layers[1])
    num_layers = num_layers or group
    ready = defaultdict(list)
    inflight = None  # (start_page, end_page, layer, layer_end)
    decide = (None, None)
    released = None  # the event that ended the D-side interval
    outcome = "open"  # released | aborted | open
    failed = False
    segments, hints = [], []

    def key(layer):
        return (layer, min(layer + group, num_layers))

    def window_ready(start, layer):
        # Same rule as the decode scheduler: a range READY in every layer group
        # loads at once (any length); otherwise a whole window x one group.
        if (
            layer == 0
            and start < prefix_pages
            and all(
                covered_until(ready[key(g)], start) > start
                for g in range(0, num_layers, group)
            )
        ):
            return True
        end = min(start + window, prefix_pages)
        return start < end and covered_until(ready[key(layer)], start) >= end

    def state():
        if failed:
            return "失败处理中"
        if inflight is not None:
            return "load进行中"
        if loaded_len // page >= prefix_pages:
            return "prefix完成·等传输/放行"
        start = loaded_len // page
        if window_ready(start, loaded_layers):
            if decide == (1, 0):
                return "本rank就绪·等其他rank"
            return "READY已到·未提交"
        return "等待READY"

    def hint():
        if inflight is None:
            return False
        start, end, layer, layer_end = inflight
        if layer_end >= num_layers:
            return end < prefix_pages and window_ready(end, 0)
        return window_ready(start, layer_end)

    last_w = plan["w"]

    def advance(w):
        if w > last_w:
            segments.append((last_w, w, state()))
            if hint():
                hints.append((last_w, w))

    for e in evs:
        w = e["w"]
        advance(w)
        last_w = w
        n = e["n"]
        if n == "notify.recv":
            (a, b), (layer, layer_end) = f_(e, "pages"), f_(e, "layers")
            ready[(layer, layer_end)] = merge_range(ready[(layer, layer_end)], a, b)
        elif n == "linker.load_submit":
            (a, b), (layer, layer_end) = f_(e, "pages"), f_(e, "layers")
            inflight = (a, b, layer, layer_end)
        elif n == "linker.load_applied":
            inflight = None
            loaded_len = f_(e, "loaded_len", loaded_len)
            loaded_layers = f_(e, "loaded_layers", loaded_layers)
            failed = failed or not f_(e, "ok", True)
        elif n == "linker.decide":
            decide = (f_(e, "local"), f_(e, "merged"))
            failed = failed or -1 in decide
            loaded_len = f_(e, "loaded_len", loaded_len)
            loaded_layers = f_(e, "loaded_layers", loaded_layers)
        elif n in ("pd.transferred", "req.abort"):
            released = e
            outcome = "released" if n == "pd.transferred" else "aborted"
            break
    open_ms = 0.0
    if released is None and cutoff is not None and cutoff > last_w:
        open_ms = (cutoff - last_w) / 1e6
        advance(cutoff)
    return dict(
        plan=plan,
        segments=segments,
        hints=hints,
        released=released,
        outcome=outcome,
        open_ms=open_ms,  # time in the last state up to the capture cutoff
        last=released or (evs[-1] if evs else plan),
        prefix_pages=prefix_pages,
        num_layers=num_layers,
    )


def state_section(states):
    totals = defaultdict(list)
    per_rid = []
    for rid, s in states.items():
        acc = defaultdict(float)
        for a, b, name in s["segments"]:
            acc[name] += (b - a) / 1e6
        hint_ms = sum((b - a) / 1e6 for a, b in s["hints"])
        total = sum(acc.values())
        for name in STATE_ORDER:
            totals[name].append(acc.get(name, 0.0))
        totals[HINT].append(hint_ms)
        per_rid.append((total, rid, acc, hint_ms, s))
    per_rid.sort(reverse=True)
    return totals, per_rid


# ------------------------------------------------------------ storage ops


def storage_section(run, role=None):
    by_thread = defaultdict(list)
    for e in run.events:
        if e["n"].startswith("od.") and role in (None, run.files[e["src"]].role):
            by_thread[(e["src"], e["th"])].append(e)
    consumed = {}
    for e in run.events:
        if e["n"] == "uc.load.consumed":
            consumed.setdefault(run.batch_key(e), e["w"])
    enqueued = {}
    for e in run.events:
        if e["n"] == "od.get.enqueue":
            enqueued[run.batch_key(e)] = e
    get = defaultdict(list)
    put = defaultdict(list)
    native = defaultdict(list)
    native_wait = defaultdict(list)
    slow_batches = []
    for (src, _), evs in by_thread.items():
        cur = None
        last_native_end = None
        for e in evs:
            n = e["n"]
            if n == "od.native":
                op = f_(e, "op")
                native[op].append(e["d"] / 1e6)
                native_wait[op].append(f_(e, "wait_us", 0) / 1e3)
                last_native_end = e["w"] + e["d"]
                if cur is not None:
                    cur["native"] += e["d"] / 1e6
            elif n == "od.get.dequeue":
                key = run.batch_key(e)
                enq = enqueued.get(key)
                cur = dict(
                    op="get",
                    batch=f_(e, "batch"),
                    key=key,
                    deq=e["w"],
                    native=0.0,
                    enq=enq,
                )
                if enq is not None:
                    get["入队→worker取出"].append((e["w"] - enq["w"]) / 1e6)
            elif n == "od.get.dep_ready" and cur is not None and cur["op"] == "get":
                cur["dep"] = e["w"]
                get["取出→前置依赖就绪"].append((e["w"] - cur["deq"]) / 1e6)
            elif n == "od.get.ranges" and cur is not None and cur["op"] == "get":
                cur["ranges"] = e["w"]
                if "dep" in cur:
                    get["依赖就绪→range构建完成"].append((e["w"] - cur["dep"]) / 1e6)
            elif n == "od.get.packed" and cur is not None and cur["op"] == "get":
                cur["packed"] = e["w"]
                if "ranges" in cur:
                    get["range构建完成→打包完成"].append((e["w"] - cur["ranges"]) / 1e6)
            elif n == "od.get.layer_ready" and cur is not None and cur["op"] == "get":
                if f_(e, "layer") == 0 and "packed" in cur and "first" not in cur:
                    cur["first"] = e["w"]
                    get["打包完成→首层完成"].append((e["w"] - cur["packed"]) / 1e6)
            elif n == "od.get.done" and cur is not None and cur["op"] == "get":
                get["native调用合计"].append(cur["native"])
                if last_native_end is not None and "dep" in cur:
                    get["最后native返回→worker观察完成"].append(
                        (e["w"] - max(last_native_end, cur["dep"])) / 1e6
                    )
                done_seen = consumed.get(cur["key"])
                if done_seen is not None:
                    get["worker观察完成→scheduler消费"].append(
                        (done_seen - e["w"]) / 1e6
                    )
                if cur["enq"] is not None:
                    total = (e["w"] - cur["enq"]["w"]) / 1e6
                    get["入队→worker观察完成"].append(total)
                    slow_batches.append((total, src, cur["batch"], cur["enq"]))
                cur = None
            elif n == "od.put.dequeue":
                cur = dict(op="put", deq=e["w"], native=0.0)
            elif n == "od.put.dep_ready" and cur is not None and cur["op"] == "put":
                put["取出→前置依赖就绪(每层)"].append((e["w"] - cur["deq"]) / 1e6)
            elif n == "od.put.group_done":
                if last_native_end is not None:
                    put["层组最后native返回→worker观察完成"].append(
                        (e["w"] - last_native_end) / 1e6
                    )
    put_enq = {}
    for e in run.events:
        key = (e["src"], f_(e, "session"), f_(e, "layer"))
        if e["n"] == "od.put.enqueue":
            put_enq[key] = e["w"]
        elif e["n"] == "od.put.dequeue" and key in put_enq:
            put["入队→worker取出(每层)"].append((e["w"] - put_enq[key]) / 1e6)
    sess = {}
    for e in run.events:
        key = (e["src"], f_(e, "session"))
        if e["n"] == "uc.save.begin":
            sess[key] = dict(begin=e["w"])
            if f_(e, "prepare_ms") is not None:
                put["session准备（调度线程CPU，forward前）"].append(f_(e, "prepare_ms"))
        elif key in sess:
            if e["n"] == "od.put.session_done":
                sess[key]["done"] = e["w"]
                put["session开始→worker观察完成"].append(
                    (e["w"] - sess[key]["begin"]) / 1e6
                )
            elif e["n"] == "uc.save.consumed" and "done" in sess[key]:
                put["session完成→scheduler回收"].append(
                    (e["w"] - sess[key]["done"]) / 1e6
                )
    slow_batches.sort(key=lambda x: x[0], reverse=True)
    return get, put, native, native_wait, slow_batches[:10]


GET_PHASES = (
    ("排队", "queue", "wait"),
    ("依赖等待", "wait", "build"),
    ("构建", "build", "submit"),
    ("提交", "submit", "io"),
    ("等完成", "io", "end"),
)


def get_phases(run):
    """Boundaries of every GET batch, from its enqueue, so a batch still queued
    or in flight is kept. Pass the full capture: a batch that started before
    the window is otherwise lost. 构建 covers ranges, packing and metadata
    upload; 提交 is the native dispatch call; 等完成 is the I/O still running
    after that call returned, until the worker sees the last layer event."""
    batches, current = {}, {}
    for e in run.events:
        n, thread = e["n"], (e["src"], e["th"])
        if n in ("od.get.enqueue", "od.get.dequeue", "od.get.dep_ready", "od.get.done"):
            batch = batches.setdefault(
                run.batch_key(e), dict(src=e["src"], queue=e["w"])
            )
            if n == "od.get.dequeue":
                batch["wait"] = e["w"]
                current[thread] = batch
            elif n == "od.get.dep_ready":
                batch["build"] = e["w"]
            elif n == "od.get.done":
                batch["end"] = e["w"]
                current.pop(thread, None)
        elif n == "od.native" and f_(e, "op") == "get" and thread in current:
            current[thread].setdefault("submit", e["w"])
            current[thread]["io"] = e["w"] + e["d"]
    return list(batches.values())


def get_spans(batch):
    """(label, start, end) of one batch. A gap between boundaries that are not
    adjacent is 未知 (events missing); the last phase of a batch that never
    finished is ·截尾 and stays open."""
    order = [start for _, start, _ in GET_PHASES] + ["end"]
    labels = {start: label for label, start, _ in GET_PHASES}
    present = [key for key in order if key in batch]
    spans = [
        (
            labels[a] if order.index(b) == order.index(a) + 1 else "未知",
            batch[a],
            batch[b],
        )
        for a, b in zip(present, present[1:])
    ]
    if present[-1] != "end":
        spans.append((f"{labels[present[-1]]}·截尾", batch[present[-1]], float("inf")))
    return spans


# ----------------------------------------------------------------- notify


def notify_section(run):
    """P send -> D receive delay, paired per receiver (``dst`` session) in
    FIFO order. A key sent by more than one P process cannot be attributed to
    one sender, so it is counted as ambiguous instead of mixing rank skew in.
    Receives with no earlier send record are counted as unmatched (the send
    is recorded before the send call, so this means missing data).
    READY causality (P side), by origin:
    * group: the offload worker calls ``on_layer_group_ready`` synchronously
      right after ``od.put.group_done``, so it pairs with the latest group
      done of the same thread and layer range;
    * receipt: paired with ``od.put.session_done`` of the same receipt only
      when the callback ran in that worker thread (fired by the completion).
      A receipt already complete at registration fires in the registering
      thread: counted, no PUT->READY latency;
    * exists: storage lookup hit, counted only.
    Returns (delays, ready, ambiguous, unmatched); ``ready`` maps a label
    to latencies (ms) or to a count."""

    def key(e):
        return (
            f_(e, "room"),
            f_(e, "dst"),
            tuple(f_(e, "pages")),
            tuple(f_(e, "layers")),
        )

    senders = defaultdict(set)
    for e in run.events:
        if e["n"] == "notify.send":
            senders[key(e)].add(e["src"])
    sent, delays = defaultdict(list), []
    ambiguous = unmatched = 0
    for e in run.events:
        if e["n"] == "notify.send":
            sent[key(e)].append(e)
        elif e["n"] == "notify.recv":
            k = key(e)
            if len(senders.get(k, ())) > 1:
                ambiguous += 1
                continue
            if not sent.get(k):
                unmatched += 1
                continue
            s = sent[k].pop(0)
            if run.files[s["src"]].host == run.files[e["src"]].host:
                delays.append((e["w"] - s["w"]) / 1e6)
    group_to_ready, receipt_to_ready = [], []
    counts = defaultdict(int)
    last_group, sessions = {}, {}
    for e in run.events:
        n, thread = e["n"], (e["src"], e["th"])
        if n == "od.put.group_done":
            last_group[thread] = (e, (f_(e, "layer_start"), f_(e, "layer_end")))
        elif n in ("od.put.dequeue", "od.put.dep_ready"):
            last_group.pop(thread, None)  # the worker moved on to the next layer
        elif n == "od.put.session_done" and f_(e, "receipt") is not None:
            sessions[(e["src"], f_(e, "receipt"))] = e
        elif n == "notify.ready":
            origin = f_(e, "origin")
            group = last_group.get(thread)
            if (
                origin in (None, "group")
                and group is not None
                and tuple(f_(e, "layers") or ()) == group[1]
            ):
                group_to_ready.append((e["w"] - group[0]["w"]) / 1e6)
            elif origin == "receipt":
                done = sessions.get((e["src"], f_(e, "receipt")))
                if done is not None and done["th"] == e["th"]:
                    receipt_to_ready.append((e["w"] - done["w"]) / 1e6)
                else:
                    counts["receipt READY（注册时已完成，不计延迟）"] += 1
            elif origin == "exists":
                counts["存储查询命中即发 READY"] += 1
            else:
                counts["来源未知 READY（旧 trace，不计延迟）"] += 1
    ready = {
        "层组READY：P 层组完成被观察→notify.ready（同线程）": group_to_ready,
        "receipt READY：PUT session完成→notify.ready（同线程回调）": receipt_to_ready,
    }
    ready.update(counts)
    return delays, ready, ambiguous, unmatched


# -------------------------------------------------------------- scheduler


def scheduler_section(run):
    stats = defaultdict(lambda: [0, 0, 0])
    for f in run.files:
        for s in f.stats:
            entry = stats[(f.role, s["n"])]
            entry[0] += s["c"]
            entry[1] += s["s"]
            entry[2] = max(entry[2], s["m"])
    batches = defaultdict(list)
    slow = []
    for e in run.events:
        if e["n"] == "sched.run_batch":
            role = run.files[e["src"]].role
            batches[(role, f_(e, "mode"))].append(e["d"] / 1e6)
        elif "d" in e and (e["n"].startswith("sched.") or e["n"] == "linker.consensus"):
            slow.append(e)
    slow.sort(key=lambda e: e["d"], reverse=True)
    return stats, batches, slow[:20]


# ---------------------------------------------------------- prefill supply


def prefill_section(run, server_round):
    """Per request on P: queue (arrive -> first EXTEND start), execution span
    (first EXTEND start -> last EXTEND end; not pure compute) and tail handoff
    (last EXTEND end -> RDMA done). Plus gaps between consecutive EXTEND
    batches of the storage owner, split by whether a request was waiting."""
    prefill = [i for i, f in enumerate(run.files) if f.role == "prefill"]
    if not prefill:
        return None
    # The owner issues PUTs; fall back to the first prefill process.
    puts = defaultdict(int)
    for e in run.events:
        if e["n"] == "od.put.enqueue" and e["src"] in prefill:
            puts[e["src"]] += 1
    owner = max(prefill, key=lambda s: (puts[s], -s))
    arrive, extends, rdma = {}, defaultdict(list), {}
    batches, exists = [], []
    tail = defaultdict(int)
    for e in run.events:
        if e["src"] != owner:
            continue
        n = e["n"]
        if n == "req.arrive" and e.get("r"):
            arrive.setdefault(e["r"], e["w"])
        elif n == "sched.run_batch" and f_(e, "mode") in ("EXTEND", "MIXED"):
            batches.append(e)
            for r in f_(e, "rids") or ():
                extends[r].append(e)
        elif n == "notify.rdma_done":
            tail["完成事件"] += 1
            rids = run.role_rids(f_(e, "room"), "prefill")
            if len(rids) == 1:
                rdma.setdefault(rids[0], e["w"])
            else:
                tail["身份冲突" if rids else "无P请求身份"] += 1
        elif n == "od.exists":
            exists.append(e)
    per_round = defaultdict(lambda: defaultdict(list))
    unmatched, conflicts = [], []
    for rid, evs in extends.items():
        start, end = evs[0]["w"], evs[-1]["w"] + evs[-1].get("d", 0)
        room = run.rid_room.get(rid)
        p_rids, d_rids = run.role_rids(room, "prefill"), run.role_rids(room, "decode")
        # server_round holds one rid per client request, so two here are two
        # clients in one room: neither round can be trusted.
        clients = {r for r in run.peer_rids(rid) if r in server_round}
        if len(clients) > 1 or len(p_rids) > 1 or len(d_rids) > 1:
            conflicts.append(
                dict(
                    rid=rid,
                    room=room,
                    prefill=p_rids,
                    decode=d_rids,
                    rounds=sorted(server_round[r] for r in clients),
                )
            )
            continue
        if server_round and not clients:
            unmatched.append(
                dict(
                    rid=rid,
                    room=room,
                    extends=len(evs),
                    tokens=sum(f_(e, "tokens", 0) for e in evs),
                    start=start,
                )
            )
            continue
        bucket = per_round[server_round[clients.pop()] if clients else "全部"]
        if rid in arrive:
            bucket["P排队"].append((start - arrive[rid]) / 1e6)
        bucket["P执行跨度"].append((end - start) / 1e6)
        if rid in rdma:
            tail["可算延迟"] += 1
            bucket["末次EXTEND结束→RDMA完成"].append((rdma[rid] - end) / 1e6)
        bucket["EXTEND次数"].append(len(evs))
    tail["关联到请求"] = sum(rid in rdma for rid in extends)
    gaps = defaultdict(list)
    gap_exists = defaultdict(list)
    for prev, nxt in zip(batches, batches[1:]):
        lo, hi = prev["w"] + prev.get("d", 0), nxt["w"]
        waiting = any(arrive.get(r, hi) < lo for r in f_(nxt, "rids") or ())
        label = "有请求在等" if waiting else "无请求在等"
        gaps[label].append((hi - lo) / 1e6)
        gap_exists[label].append(
            sum(x.get("d", 0) for x in exists if lo <= x["w"] < hi) / 1e6
        )
    admitted = len(extends) - len(unmatched) - len(conflicts)
    exists_parts = defaultdict(list)
    for e in exists:
        for field, label in (
            ("put_lock_ms", "等PUT锁"),
            ("put_sync_ms", "等PUT流同步"),
            ("route_ms", "路由与建metadata"),
            ("upload_ms", "metadata上传（含等当前流）"),
            ("native_ms", "native查询"),
        ):
            if f_(e, field) is not None:
                exists_parts[label].append(f_(e, field))
    return dict(
        owner=run.files[owner].label,
        per_round=per_round,
        gaps=gaps,
        gap_exists=gap_exists,
        exists_count=len(exists),
        exists_parts=exists_parts,
        admitted=admitted,
        tail=tail,
        unmatched=unmatched,
        conflicts=conflicts,
    )


# --------------------------------------------------- decode step intervals


def decode_step_section(run, full=None):
    """Start-to-start intervals between consecutive DECODE batches of one D
    process. Classified by explicit state, never by a length threshold:
    an IDLE batch in between -> 含空闲; else shared rids -> 持续decode;
    else 无法判定 (kept). This is a scheduling interval, not per-request
    TPOT, and the groups show correlation only. GET lifetimes come from
    ``full`` so one started before the window still counts as in flight."""
    by_src = defaultdict(list)
    gets = defaultdict(list)
    consensus = defaultdict(list)
    for e in run.events:
        if run.files[e["src"]].role != "decode":
            continue
        n = e["n"]
        if n == "sched.run_batch" and f_(e, "mode") in ("DECODE", "IDLE"):
            by_src[e["src"]].append(e)
        elif n == "linker.consensus" and "d" in e:
            consensus[e["src"]].append(e["w"])
    prefill_gets = []
    for batch in get_phases(full or run):
        span = (batch["queue"], batch.get("end", float("inf")))
        if run.files[batch["src"]].role == "decode":
            gets[batch["src"]].append(span)
        else:
            prefill_gets.append(span)
    kinds = defaultdict(list)
    groups = defaultdict(list)
    steps = []
    for src, evs in by_src.items():
        prev, idle = None, False
        for e in evs:
            if f_(e, "mode") == "IDLE":
                idle = prev is not None
                continue
            if prev is not None:
                interval = (e["w"] - prev["w"]) / 1e6
                a, b = set(f_(prev, "rids") or ()), set(f_(e, "rids") or ())
                if idle:
                    kinds["含空闲"].append(interval)
                elif a & b:
                    kinds["持续decode"].append(interval)
                    lo, hi = prev["w"], e["w"]
                    steps.append((src, lo, hi))
                    get = any(s < hi and t > lo for s, t in gets[src])
                    other = any(
                        s < hi and t > lo
                        for peer, spans in gets.items()
                        if peer != src
                        for s, t in spans
                    )
                    if not get and not other:
                        busy = any(s < hi and t > lo for s, t in prefill_gets)
                        groups["D无GET·P有GET在途" if busy else "D无GET·P无GET"].append(
                            interval
                        )
                    slow = any(lo <= w < hi for w in consensus[src])
                    groups["本进程GET在途" if get else "本进程无GET在途"].append(
                        interval
                    )
                    groups["其他D有GET在途" if other else "其他D无GET在途"].append(
                        interval
                    )
                    groups["有≥2ms consensus" if slow else "无≥2ms consensus"].append(
                        interval
                    )
                    groups["有新请求加入" if b - a else "无新请求加入"].append(interval)
                else:
                    kinds["无法判定"].append(interval)
            prev, idle = e, False
    return kinds, groups, steps


def dp_group(run, src):
    match = re.search(r"DP(\d+)", run.files[src].label)
    return match.group(1) if match else None


def covered(spans, lo, hi):
    """Time in [lo, hi] covered by the union of ``spans``."""
    total, reach = 0, lo
    for start, stop in sorted(spans):
        start, stop = max(start, reach), min(stop, hi)
        if stop > start:
            total += stop - start
            reach = stop
    return total


SPIKE_UNRANKED = (
    "copy_done",
    "其他copy_done",
    "P",
    "P保存session",
    "GPU forward",
    "GPU GET",
)


def decode_spike_section(run, steps, top=20):
    """The longest DECODE intervals with what overlapped them on this host.
    ``run`` is the full capture, so activity that began before the window is
    kept; ``steps`` carries the window. One stall appears on every rank, so
    intervals of comparable length that overlap are one spike. Same-kind
    spans of one process are merged, and each cell is the largest such
    coverage. Only the representative rank's DECODE wait is copy_done; waits
    elsewhere are listed apart, and release/output/evict are limited to the
    spike's ranks. A P save session is a lifetime spanning forward and queues,
    not PUT execution. P GETs are shown but not ranked, so they cannot hide a D
    GET. With SGLANG_UCL_TRACE_GPU the GPU columns show device time: forward
    coverage near the interval means a forward was running through the stall;
    near one frame means it had not started. Overlap is correlation only, so
    ``top`` is the largest overlapping D activity and not a cause."""
    if run.cross_host or not steps:
        return None
    spikes = []
    for src, lo, hi in sorted(steps, key=lambda s: s[1] - s[2]):
        same = next((x for x in spikes if lo < x["hi"] and hi > x["lo"]), None)
        if same is None:
            if len(spikes) < top:
                spikes.append(dict(src=src, lo=lo, hi=hi, ranks={src}))
        elif (hi - lo) * 2 >= same["hi"] - same["lo"]:
            same["ranks"].add(src)
    activities = []  # (src, column or None for GET scope, label, start, end)
    for batch in get_phases(run):
        for label, start, stop in get_spans(batch):
            activities.append((batch["src"], None, label, start, stop))
    columns = {
        "sched.copy_done": "copy_done",
        "sched.finish_release": "释放",
        "sched.stream_output": "输出",
        "pd.evict": "驱逐",
        "gpu.forward": "GPU forward",
        "gpu.get": "GPU GET",
    }
    sessions = {}
    for e in run.events:
        role = run.files[e["src"]].role
        if e["n"] in columns and role == "decode" and "d" in e:
            label = f_(e, "mode", "") if e["n"] == "sched.copy_done" else ""
            activities.append(
                (e["src"], columns[e["n"]], label, e["w"], e["w"] + e["d"])
            )
        elif e["n"] == "uc.save.begin":
            sessions[(e["src"], f_(e, "session"))] = e["w"]
        elif e["n"] == "od.put.session_done":
            begin = sessions.pop((e["src"], f_(e, "session")), None)
            if begin is not None:
                activities.append((e["src"], "P保存session", "", begin, e["w"]))
    for (src, _), begin in sessions.items():
        activities.append((src, "P保存session", "截尾", begin, float("inf")))

    def cell_of(spike, src, column, label):
        """The (column, label) of one activity, None when out of scope."""
        if column == "copy_done":
            if src == spike["src"] and label == "DECODE":
                return column, ""
            scope = "参与" if src in spike["ranks"] else "其他"
            return "其他copy_done", scope + label
        if column == "P保存session":
            return column, label
        if column == "GPU GET":
            return column, "本进程" if src == spike["src"] else "其他D"
        if column is not None:
            return (column, label) if src in spike["ranks"] else None
        if run.files[src].role != "decode":
            return "P", label
        if src == spike["src"]:
            return "本进程GET", label
        group = dp_group(run, src)
        same = group is not None and group == dp_group(run, spike["src"])
        return ("同组GET" if same else "其他组GET"), label

    for spike in spikes:
        spans = defaultdict(list)
        for src, column, label, start, stop in activities:
            if start < spike["hi"] and stop > spike["lo"]:
                cell = cell_of(spike, src, column, label)
                if cell is not None:
                    spans[(*cell, src)].append((start, stop))
        cells = defaultdict(float)
        for (name, label, _), value in spans.items():
            ms = covered(value, spike["lo"], spike["hi"]) / 1e6
            cells[(name, label)] = max(cells[(name, label)], ms)
        spike["cells"] = cells
        ranked = {k: v for k, v in cells.items() if k[0] not in SPIKE_UNRANKED}
        name, label = max(ranked, key=ranked.get) if ranked else ("无", "")
        spike["top"] = f"{name}·{label}" if label else name
        spike["ms"] = (spike["hi"] - spike["lo"]) / 1e6
        spike["copy"] = cells.get(("copy_done", ""), 0.0)
    return spikes


def spike_cell(spike, column):
    parts = [
        f"{label + ' ' if label else ''}{value:.0f}"
        for (name, label), value in sorted(
            spike["cells"].items(), key=lambda item: -item[1]
        )
        if name == column
    ]
    return "；".join(parts) or "-"


STREAM_NAMES = ("forward", "schedule", "copy", "get", "put")


def stream_section(run):
    """CUDA streams per process from the startup record. Two names on one
    handle are the same stream, so the second serializes behind the first.
    Different handles can still share a hardware queue, which no event shows."""
    streams, limits = defaultdict(dict), {}
    for e in run.events:
        if e["n"] == "gpu.streams":
            streams[e["src"]].update(f_(e, "streams") or {})
            limits[e["src"]] = f_(e, "max_connections")
    rows, aliased = [], []
    for src in sorted(streams):
        by_handle = defaultdict(list)
        for name in STREAM_NAMES:
            if name in streams[src]:
                by_handle[streams[src][name][1]].append(name)
        same = [names for names in by_handle.values() if len(names) > 1]
        aliased += [(src, names) for names in same]
        rows.append(
            [
                run.files[src].label,
                limits[src],
                *(streams[src].get(name, ["-", "-"])[0] for name in STREAM_NAMES),
                "；".join("=".join(names) for names in same) or "无",
            ]
        )
    return rows, aliased


def decode_cache_section(run, server_round):
    """Per round: D prefix hit at admission and logical pages to restore,
    from one representative rank per request (page ranges merged across layer
    groups), and GET work over all ranks in page-layers, which is not a miss
    count. Evictions per DP group, from its lowest rank, by round window; a
    time inside two windows is listed apart instead of being counted twice."""
    first, need, work, evicts = {}, defaultdict(list), defaultdict(int), []
    tags, windows = {}, {}
    for e in run.events:
        n, src = e["n"], e["src"]
        if n == "client.send":
            tags[e["r"]] = str(f_(e, "tag", e["r"])).split("-", 1)[0]
            lo, hi = windows.get(tags[e["r"]], (e["w"], e["w"]))
            windows[tags[e["r"]]] = (min(lo, e["w"]), hi)
        elif n == "client.end" and e.get("r") in tags:
            lo, hi = windows[tags[e["r"]]]
            windows[tags[e["r"]]] = (lo, max(hi, e["w"]))
        elif run.files[src].role != "decode":
            continue
        elif n == "pd.prealloc" and e.get("r") not in first:
            first[e["r"]] = (
                src,
                f_(e, "local_prefix", 0),
                f_(e, "decode_prefix_len", 0),
            )
        elif n == "linker.load_submit":
            rid = run.rid_of(e)
            pages, layers = f_(e, "pages"), f_(e, "layers")
            work[rid] += (pages[1] - pages[0]) * (layers[1] - layers[0])
            if rid in first and first[rid][0] == src:
                need[rid] = merge_range(need[rid], *pages)
        elif n == "pd.evict":
            evicts.append(e)
    rounds = defaultdict(lambda: defaultdict(list))
    for rid, (_, hit, prefix) in first.items():
        bucket = rounds[server_round.get(rid, "未匹配")]
        bucket["hit"].append(hit)
        bucket["prefix"].append(prefix)
        bucket["need"].append(sum(b - a for a, b in need[rid]))
        bucket["work"].append(work[rid])
    leaders = {}
    for e in evicts:
        group = dp_group(run, e["src"])
        leaders[group] = min(leaders.get(group, e["src"]), e["src"])
    evicted = defaultdict(lambda: [0, 0, 0.0])
    for e in evicts:
        group = dp_group(run, e["src"])
        if e["src"] != leaders[group]:
            continue
        inside = [r for r, (lo, hi) in windows.items() if lo <= e["w"] <= hi]
        where = inside[0] if len(inside) == 1 else ("交叠" if inside else "轮次外")
        row = evicted[(where, group)]
        row[0] += 1
        row[1] += f_(e, "freed", 0)
        row[2] += e.get("d", 0) / 1e6
    return rounds, evicted


# ----------------------------------------------------------------- report


def build_report(run):
    lines = [
        f"# UCLinker trace report: {os.path.basename(os.path.normpath(run.dir))}",
        "",
    ]
    clock = (
        "跨机：各主机内部用同一 perf 时钟；主机间按各自 wall 时钟锚点粗对齐，"
        "未校准，跨机差值和重叠不作结论。"
        if run.cross_host
        else "单机：所有进程共用同一 perf 时钟（CLOCK_MONOTONIC），可直接比较。"
    )
    span_ms = run.ms(run.events[-1]["w"]) if run.events else 0
    lines += [
        f"- 主机：{', '.join(run.hosts)}；时钟域：{clock}",
        f"- 事件：{len(run.events)}，时间跨度 {span_ms / 1000:.1f} s",
        "",
    ]
    summary = {}

    rows, problems, unclosed = completeness(run)
    lines += ["## 1. 数据完整性", ""]
    lines += [
        table(["角色", "进程", "主机", "pid", "事件数", "最后刷盘 s", "状态"], rows),
        "",
    ]
    if unclosed:
        lines += [
            "未闭合的操作（开始有记录、结束没有；可能仍在进行、卡住或丢失）：",
            "",
        ]
        lines += [table(["开始事件", "结束事件", "数量"], unclosed), ""]
    lines += [f"- {p}" for p in problems] or ["- 未发现丢弃或缺失的结束标记。"]
    lines += [""]
    summary["incomplete"] = bool(problems or unclosed)

    crows, csum, cdetail, has_client, server_round, via_sse = client_section(run)
    lines += ["## 2. 客户端（毫秒）", ""]
    if has_client:
        lines += [
            "首帧 = 第一个 SSE data 帧（rounds.json 旧 TTFT 口径）；首内容 = 第一个带 content 的帧；"
            "帧间隔是相邻带内容帧的间隔，不等于逐 token TBT。",
            "",
            table(
                [
                    "轮",
                    "成功",
                    "首帧 p50",
                    "首内容 p50",
                    "首内容 max",
                    "帧间隔 p50",
                    "帧间隔 p99",
                    "帧间隔 max",
                    "端到端 p50",
                    "rid 匹配服务端",
                ],
                crows,
            ),
            "",
        ]
        if via_sse:
            lines += [
                f"- {via_sse} 个请求的发送 rid 未被服务端采用（router 改写或丢弃），"
                "按 SSE 帧的 id（服务端 rid）关联。",
                "",
            ]
    else:
        lines += ["没有客户端事件（客户端未开启 SGLANG_UCL_TRACE）。", ""]
    summary["client"] = csum

    # Server-side sections cover only the client experiment window; earlier
    # check/warm-up traffic is listed separately below instead of mixed in.
    sends = [e["w"] for e in run.events if e["n"] == "client.send"]
    ends = [e["w"] for e in run.events if e["n"] == "client.end"]
    full = run
    if sends and ends:
        run = run.windowed(min(sends), max(ends))
        lo, hi = run.window
        lines += [
            f"以下第 3–8 节只统计客户端实验窗口 {full.ms(lo) / 1000:.1f}–"
            f"{full.ms(hi) / 1000:.1f} s 内的事件；窗口外（预热/check）见第 9 节。"
            "第 6 节的阶段累计表来自进程全程统计，不受窗口限制。",
            "",
        ]

    states = decode_states(run)
    for rid, s in states.items():
        if s["window_cut"]:  # what the full capture says happened afterwards
            s["final"] = next(
                (
                    e["n"]
                    for e in full.events
                    if e["w"] > run.window[1]
                    and e["n"] in ("pd.transferred", "req.abort")
                    and full.rid_of(e) == rid
                ),
                None,
            )
    totals, per_rid = state_section(states)
    lines += ["## 3. D 侧请求状态（毫秒）", ""]
    if states:
        lines += [
            "状态互斥，按事件逐段划分，加起来等于从 D 预分配到放行（或 abort）的时间；"
            "未结束的请求停在最后状态，计到该进程最后一次刷盘（仍在进行），"
            "有实验窗口时最多计到窗口末尾。"
            "并行进度看 trace.json 的“请求并行轨道”。最后一列是提示，可与 load进行中 重叠。",
            "",
        ]
        rows = []
        for name in STATE_ORDER + (HINT,):
            d = dist(totals[name])
            rows.append([name, fmt(d["sum"]), fmt(d["p50"]), fmt(d["max"])])
        lines += [table(["状态", "合计", "每请求 p50", "每请求 max"], rows), ""]
        lines += ["最慢 10 个请求：", ""]
        rows = []
        for total, rid, acc, hint_ms, s in per_rid[:10]:
            end = outcome_text(s)
            parts = "，".join(f"{k} {acc[k]:.0f}" for k in STATE_ORDER if acc.get(k))
            first = cdetail.get(rid, {}).get("first_content_ms")
            rows.append([rid, fmt(total), parts, fmt(hint_ms), fmt(first), end])
        lines += [
            table(["rid", "D 侧合计", "分解", "提示", "客户端首内容", "结局"], rows),
            "",
        ]
    else:
        lines += ["没有 linker.plan 事件（D 未开启 notify 或未开启 trace）。", ""]
    summary["decode_states"] = {k: dist(v) for k, v in totals.items()}

    get, put, native, native_wait, slow_batches = storage_section(run)
    lines += [
        "## 4. OceanDisk 操作（毫秒）",
        "",
        "“worker观察完成”是 worker 线程等到 CUDA event 后的时间，不等于设备完成时间。",
        "",
    ]
    rows = [
        [f"GET {name} {k}", *dist_cells(v)]
        for role, name in (("prefill", "P"), ("decode", "D"))
        for k, v in storage_section(run, role)[0].items()
    ] or [["GET " + k, *dist_cells(v)] for k, v in get.items()]
    rows += [["PUT " + k, *dist_cells(v)] for k, v in put.items()]
    rows += [[f"native {op} 调用", *dist_cells(v)] for op, v in native.items()]
    rows += [
        [f"native {op} 进入前等待（锁/上下文）", *dist_cells(v)]
        for op, v in native_wait.items()
    ]
    if rows:
        lines += [table(["阶段", "次数", "p50", "p99", "max"], rows), ""]
    if slow_batches:
        lines += ["最慢 10 个 GET batch：", ""]
        lines += [
            table(
                ["入队→观察完成", "进程", "batch", "tags", "页数"],
                [
                    [
                        fmt(t),
                        run.files[src].label,
                        b,
                        len(f_(enq, "tags", [])),
                        sum(f_(enq, "pages", [])),
                    ]
                    for t, src, b, enq in slow_batches
                ],
            ),
            "",
        ]
    summary["get"] = {k: dist(v) for k, v in get.items()}
    summary["put"] = {k: dist(v) for k, v in put.items()}

    delays, ready, ambiguous, unmatched = notify_section(run)
    lines += ["## 5. notify（毫秒）", ""]
    rows = [
        [label, *dist_cells(v)] for label, v in ready.items() if isinstance(v, list)
    ]
    rows += [
        [label, v, "-", "-", "-"] for label, v in ready.items() if isinstance(v, int)
    ]
    if run.cross_host:
        rows.append(["P 发送→D 接收", "跨机未校准，不计算", "", "", ""])
    else:
        rows.append(
            ["P 发送→D 接收（含发送调用，同一接收方按顺序配对）", *dist_cells(delays)]
        )
    lines += [table(["阶段", "次数", "p50", "p99", "max"], rows), ""]
    if ambiguous:
        lines += [
            f"- {ambiguous} 条接收无法归属到唯一发送进程（多个 P 进程向同一接收方"
            "发送了相同范围），未计入传递延迟。",
            "",
        ]
    if unmatched:
        lines += [
            f"- {unmatched} 条接收找不到更早的发送记录（P 侧事件丢失或 P 未开 trace），"
            "未计入传递延迟。",
            "",
        ]

    stats, batches, slow = scheduler_section(run)
    lines += [
        "## 6. 调度",
        "",
        "run_batch 是 CPU 侧耗时（overlap 模式下主要是提交），不是 GPU 执行区间。",
        "",
    ]
    rows = [[role, mode, *dist_cells(v)] for (role, mode), v in sorted(batches.items())]
    if rows:
        lines += [table(["角色", "batch 类型", "次数", "p50", "p99", "max"], rows), ""]
    rows = [
        [role, name, c, fmt(s / 1e6), fmt(s / c / 1e3 if c else 0), fmt(m / 1e6, 2)]
        for (role, name), (c, s, m) in sorted(stats.items())
    ]
    if rows:
        lines += [
            "阶段累计（进程全程，含窗口外；合计是各次调用耗时之和，含等待及各 rank 重叠，"
            "不等于可省下的 CPU 时间）：",
            "",
            table(["角色", "阶段", "调用次数", "合计 ms", "平均 us", "max ms"], rows),
            "",
        ]
    if slow:
        lines += ["最慢的调度调用（超过阈值才单独记录）：", ""]
        lines += [
            table(
                ["时间 s", "进程", "名称", "耗时 ms"],
                [
                    [
                        fmt(run.ms(e["w"]) / 1000, 3),
                        run.files[e["src"]].label,
                        e["n"],
                        fmt(e["d"] / 1e6, 2),
                    ]
                    for e in slow
                ],
            ),
            "",
        ]

    p = prefill_section(run, server_round)
    lines += ["## 7. P 侧供给（毫秒）", ""]
    if p is None:
        lines += ["没有 prefill 进程的 trace。", ""]
    else:
        lines += [
            f"按存储 owner 进程 {p['owner']} 统计。执行跨度 = 首个 EXTEND 开始→最后一个 "
            "EXTEND 结束，含等待 GET、PUT 准备等，不是纯计算时间。",
            "",
        ]
        rows = []
        for rnd in sorted(p["per_round"], key=lambda r: (len(r), r)):
            b = p["per_round"][rnd]
            for name in ("P排队", "P执行跨度", "末次EXTEND结束→RDMA完成", "EXTEND次数"):
                if b.get(name):
                    rows.append([rnd, name, *dist_cells(b[name])])
        lines += [table(["轮", "阶段", "次数", "p50", "p99", "max"], rows), ""]
        tail = p["tail"]
        lines += [
            f"- RDMA 完成事件 {tail['完成事件']}，关联到 P 请求 {tail['关联到请求']}，"
            f"其中可算 tail 延迟 {tail['可算延迟']}；身份冲突 {tail['身份冲突']}，"
            f"无 P 请求身份 {tail['无P请求身份']}。",
            "",
        ]
        if p["unmatched"]:
            lines += [
                "未匹配到本次客户端的 P 请求（不计入上表和比值的分母）：",
                "",
                table(
                    ["rid", "room", "EXTEND次数", "tokens", "开始 s"],
                    [
                        [
                            u["rid"],
                            u["room"],
                            u["extends"],
                            u["tokens"],
                            fmt(run.ms(u["start"]) / 1000, 3),
                        ]
                        for u in p["unmatched"]
                    ],
                ),
                "",
            ]
        if p["conflicts"]:
            lines += [
                "身份冲突的 P 请求（同一 room 对应多个 P/D 身份或多个客户端请求，"
                "无法确定归属，不计入上表和比值的分母）：",
                "",
                table(
                    ["rid", "room", "P 身份", "D 身份", "客户端轮次"],
                    [
                        [
                            c["rid"],
                            c["room"],
                            ",".join(c["prefill"]),
                            ",".join(c["decode"]),
                            ",".join(c["rounds"]) or "-",
                        ]
                        for c in p["conflicts"]
                    ],
                ),
                "",
            ]
        rows = []
        for label in ("有请求在等", "无请求在等"):
            if p["gaps"].get(label):
                rows.append([label, *dist_cells(p["gaps"][label])])
                rows.append(
                    [
                        f"{label}·其中 od.exists 耗时",
                        *dist_cells(p["gap_exists"][label]),
                    ]
                )
        if rows:
            lines += [
                "相邻 EXTEND 之间的间隙（前一批结束→后一批开始；“有请求在等”= 后一批"
                "有请求在前一批结束前已到达）：",
                "",
                table(["间隙", "次数", "p50", "p99", "max"], rows),
                "",
            ]
        if p["exists_parts"]:
            lines += [
                "od.exists 分段：",
                "",
                table(
                    ["分段", "次数", "p50", "p99", "max"],
                    [[k, *dist_cells(v)] for k, v in p["exists_parts"].items()],
                ),
                "",
            ]
        ratio = p["exists_count"] / p["admitted"] if p["admitted"] else 0
        lines += [
            f"- owner 全窗口 od.exists 查询数 {p['exists_count']}（无请求身份，"
            f"含未匹配请求的查询，无法逐请求过滤），匹配客户端的请求数 {p['admitted']}，"
            f"比值 {ratio:.2f}。只作排查信号：一个请求正常就会查询两次"
            "（handoff 前缀页、match 剩余前缀），比值大于 1 不等于重复查询；"
            "实施 C2 前须确认重复的是同一请求、同一查询区间。",
            "",
        ]
        summary["prefill"] = {
            rnd: {k: dist(v) for k, v in b.items()} for rnd, b in p["per_round"].items()
        }
        summary["prefill_exists_ratio"] = ratio

    kinds, groups, steps = decode_step_section(run, full)
    lines += ["## 8. D 连续 decode 调度间隔（毫秒）", ""]
    if kinds:
        lines += [
            "相邻 DECODE batch 的提交起点间隔，是调度间隔而不是逐请求 TPOT；"
            "按显式状态划分（中间有 IDLE → 含空闲；两批有相同 rid → 持续decode；"
            "其余 → 无法判定），不按时长剔除。分组只表示相关，因果须做同负载 A/B；"
            "consensus 只有 ≥2ms 的调用才逐次记录。",
            "",
        ]
        rows = [
            [k, *dist_cells(kinds[k])] for k in ("持续decode", "含空闲", "无法判定")
        ]
        rows += [[f"持续decode·{k}", *dist_cells(v)] for k, v in sorted(groups.items())]
        lines += [table(["样本", "次数", "p50", "p99", "max"], rows), ""]
        summary["decode_steps"] = {k: dist(v) for k, v in {**kinds, **groups}.items()}
    else:
        lines += ["没有 DECODE batch 记录。", ""]

    spikes = decode_spike_section(full, steps)
    lines += ["## 8.1 D 尖峰与同时段活动（毫秒）", ""]
    if spikes is None:
        lines += ["跨机时钟未校准或没有持续 decode 样本，不计算。", ""]
    else:
        lines += [
            "最长的持续 decode 间隔；多个 rank 上时长相近且重叠的间隔算一个尖峰。"
            "同进程同类区间先求并集，单元格取单个进程的最大覆盖时长；"
            "只表示同时发生，“最大重叠活动”也不是根因。活动取自完整采集，不受窗口裁剪。"
            "copy_done 只算“进程”列这个代表进程的 DECODE 等待，包含等待 forward 及其依赖；"
            "其他 rank 或 IDLE 的等待列在“其他copy_done”；释放、输出、驱逐只看参与尖峰的 rank。"
            "P保存session 是 uc.save.begin→session 完成的存续时间，含 forward 和排队，不是 PUT 执行。"
            "P、P保存session、两个 copy_done 列和两个 GPU 列不参与最大重叠活动的排名。"
            "GPU 列需要 SGLANG_UCL_TRACE_GPU=1：GPU forward 接近间隔表示 forward 在 GPU 上跑满了停顿，"
            "接近一帧表示停顿期间 forward 尚未开始；GPU GET 是 GET 在其 GPU 流上的时间。"
            "构建含打包和元数据上传；提交是 native 调用；等完成是调用返回后 I/O 仍在途的时间；"
            "“·截尾”表示采集结束时仍未完成，“未知”表示中间事件缺失。",
            "",
        ]
        tally = defaultdict(lambda: [0, 0])
        for spike in spikes:
            tally[spike["top"]][0] += 1
            tally[spike["top"]][1] += int(spike["copy"] * 2 >= spike["ms"])
        lines += [
            table(
                ["最大重叠活动", "尖峰数", "其中代表进程 copy_done ≥ 间隔一半"],
                [
                    [k, a, b]
                    for k, (a, b) in sorted(tally.items(), key=lambda x: -x[1][0])
                ],
            ),
            "",
            table(
                [
                    "时间 s",
                    "间隔",
                    "rank数",
                    "进程",
                    "本进程GET",
                    "同组GET",
                    "其他组GET",
                    "P",
                    "P保存session",
                    "GPU forward",
                    "GPU GET",
                    "copy_done",
                    "其他copy_done",
                    "释放",
                    "输出",
                    "驱逐",
                ],
                [
                    [
                        fmt(run.ms(x["lo"]) / 1000, 3),
                        fmt(x["ms"]),
                        len(x["ranks"]),
                        run.files[x["src"]].label,
                        *(
                            spike_cell(x, column)
                            for column in (
                                "本进程GET",
                                "同组GET",
                                "其他组GET",
                                "P",
                                "P保存session",
                                "GPU forward",
                                "GPU GET",
                                "copy_done",
                                "其他copy_done",
                                "释放",
                                "输出",
                                "驱逐",
                            )
                        ),
                    ]
                    for x in spikes
                ],
            ),
            "",
        ]
        stop = run.window[1] if run.window else float("inf")
        pending = [
            b
            for b in get_phases(full)
            if run.files[b["src"]].role == "decode"
            and b["queue"] <= stop
            and b.get("end", float("inf")) > stop
        ]
        queued = sum(b.get("wait", float("inf")) > stop for b in pending)
        lines += [
            f"- 统计结束时 D 侧 GET：仍在排队 {queued} 批，"
            f"已取出未完成 {len(pending) - queued} 批。",
            "",
        ]
        summary["decode_spikes"] = {k: v[0] for k, v in tally.items()}

    rows, aliased = stream_section(full)
    lines += ["## 8.3 GPU 流身份", ""]
    if rows:
        lines += [
            "启动时记录的 stream_id。重合列为同一条流，后者必然排在前者之后；"
            "不重合也可能共用硬件队列，这需要对比不同 CUDA_DEVICE_MAX_CONNECTIONS 的结果。",
            "",
            table(["进程", "连接上限", *STREAM_NAMES, "重合"], rows),
            "",
        ]
        summary["stream_aliases"] = len(aliased)
    else:
        lines += ["没有 gpu.streams 记录。", ""]

    rounds, evicted = decode_cache_section(run, server_round)
    lines += ["## 8.2 D 命中与恢复量", ""]
    if rounds:
        lines += [
            "命中和需恢复页取每个请求的一个代表 rank，需恢复页跨层组去重；"
            "GET 工作量是全部 rank 的“页×层”之和，含层组重复和 rank 重复，不是未命中数。",
            "",
            table(
                [
                    "轮",
                    "请求数",
                    "准入命中 token p50",
                    "前缀 token p50",
                    "需恢复逻辑页 p50",
                    "GET 页×层 合计",
                ],
                [
                    [
                        rnd,
                        len(b["hit"]),
                        fmt(pct(b["hit"], 50), 0),
                        fmt(pct(b["prefix"], 50), 0),
                        fmt(pct(b["need"], 50), 0),
                        sum(b["work"]),
                    ]
                    for rnd, b in sorted(
                        rounds.items(), key=lambda x: (len(x[0]), x[0])
                    )
                ],
            ),
            "",
        ]
    else:
        lines += ["没有 pd.prealloc 事件。", ""]
    if evicted:
        lines += [
            "驱逐（每个 DP 组取一个 rank，按轮次时间窗统计，不归给单个请求）：",
            "",
            table(
                ["轮", "DP组", "次数", "实际释放 token", "耗时合计 ms"],
                [
                    [where, group, a, b, fmt(c)]
                    for (where, group), (a, b, c) in sorted(
                        evicted.items(), key=lambda x: (len(x[0][0]), str(x[0]))
                    )
                ],
            ),
            "",
        ]

    if full.window is None and run.window is not None:
        lo, hi = run.window
        outside = [e for e in full.events if not lo <= e["w"] <= hi]
        counts = defaultdict(int)
        for e in outside:
            if e["n"] == "sched.run_batch":
                counts[(full.files[e["src"]].role, f_(e, "mode"))] += 1
        slow_out = sorted(
            (
                e
                for e in outside
                if "d" in e
                and (e["n"].startswith("sched.") or e["n"] == "linker.consensus")
            ),
            key=lambda e: e["d"],
            reverse=True,
        )[:10]
        lines += ["## 9. 实验窗口外（预热/check）", ""]
        if counts:
            lines += [
                table(
                    ["角色", "batch 类型", "次数"],
                    [[r, m, c] for (r, m), c in sorted(counts.items())],
                ),
                "",
            ]
        if slow_out:
            lines += [
                table(
                    ["时间 s", "进程", "名称", "耗时 ms"],
                    [
                        [
                            fmt(full.ms(e["w"]) / 1000, 3),
                            full.files[e["src"]].label,
                            e["n"],
                            fmt(e["d"] / 1e6, 2),
                        ]
                        for e in slow_out
                    ],
                ),
                "",
            ]
        if not counts and not slow_out:
            lines += ["无。", ""]

    lines += ["## 10. 未完成与异常", ""]
    issues = []
    for rid, s in states.items():
        if s["outcome"] == "open":
            where = "窗口结束时" if s["window_cut"] else ""
            issues.append(
                [
                    rid,
                    (
                        f"D {where}未放行（停在 {s['segments'][-1][2]} "
                        f"{s['open_ms']:.0f} ms，{final_text(s)}）"
                        if s["open_ms"] and s["segments"]
                        else f"D {where}未放行"
                    ),
                    s["last"]["n"],
                    fmt(run.ms(s["last"]["w"]) / 1000, 3),
                ]
            )
    for e in full.events:  # failures outside the window still matter
        if e["n"] in ("req.abort", "notify.fail", "notify.recv_invalid") or (
            e["n"] in ("od.get.done", "linker.load_applied", "client.end")
            and f_(e, "ok") is False
        ):
            issues.append(
                [
                    run.rid_of(e) or "-",
                    e["n"],
                    str(f_(e, "reason") or f_(e, "error") or ""),
                    fmt(run.ms(e["w"]) / 1000, 3),
                ]
            )
    if issues:
        lines += [table(["rid", "情况", "最后事件/原因", "时间 s"], issues[:50]), ""]
    else:
        lines += ["无。", ""]
    return "\n".join(lines) + "\n", summary


def outcome_text(s):
    if s["outcome"] == "released":
        return "已放行"
    if s["outcome"] == "aborted":
        return "abort"
    tail = (
        f"，最后状态已持续 {s['open_ms']:.0f}（{final_text(s)}）"
        if s["open_ms"]
        else ""
    )
    where = "窗口结束时" if s.get("window_cut") else ""
    return f"{where}未放行，最后事件 {s['last']['n']}{tail}"


def final_text(s):
    if not s.get("window_cut"):
        return "仍在进行"
    return {
        "pd.transferred": "全程采集中随后放行",
        "req.abort": "全程采集中随后 abort",
    }.get(s.get("final"), "全程采集结束仍未放行")


def dist_cells(values):
    d = dist(values)
    return [d["n"], fmt(d["p50"], 2), fmt(d["p99"], 2), fmt(d["max"], 2)]


# --------------------------------------------------------------- timeline


def build_trace(run):
    out = []
    us = lambda w: (w - run.t0) / 1000  # noqa: E731
    for index, f in enumerate(run.files):
        out.append(dict(ph="M", name="process_name", pid=index, args=dict(name=f.name)))
    for e in run.events:
        args = dict(e.get("f") or {})
        if e.get("r"):
            args["rid"] = e["r"]
        item = dict(
            name=e["n"], pid=e["src"], tid=str(e["th"]), ts=us(e["w"]), args=args
        )
        if "d" in e:
            item.update(ph="X", dur=e["d"] / 1000)
        else:
            item.update(ph="i", s="t")
        out.append(item)
    out += request_tracks(run, decode_states(run), len(run.files), us)
    return dict(traceEvents=out, displayTimeUnit="ms")


LANES = (
    "客户端",
    "P 调度",
    "P PUT",
    "P READY",
    "D READY",
    "D 状态",
    "D 提示",
    "D load",
    "D 传输/放行",
    "D 调度",
)


def request_tracks(run, states, pid, us):
    """One group of lanes per request, joined through rid, room, load tag,
    GET batch and PUT session, so READY, load, transfer and release of
    concurrent requests line up without hunting across threads."""
    role = lambda e: run.files[e["src"]].role  # noqa: E731
    owner = {}  # (rid, role) -> representative src (first scheduler seen)
    first = {}
    for e in run.events:
        rid = run.rid_of(e)
        if rid is None:
            for r in f_(e, "rids") or ():
                first.setdefault(r, e["w"])
            continue
        first.setdefault(rid, e["w"])
        if e["n"] in ("req.arrive", "linker.plan"):
            owner.setdefault((rid, role(e)), e["src"])
    for rid, s in states.items():
        owner[(rid, "decode")] = s["plan"]["src"]
    order = {rid: i for i, rid in enumerate(sorted(first, key=first.get))}
    lane_of = {name: i for i, name in enumerate(LANES)}
    out, used = [], set()

    def tid(rid, lane):
        t = order[rid] * len(LANES) + lane_of[lane] if rid in order else None
        if t is not None and t not in used:
            used.add(t)
            out.append(
                dict(
                    ph="M",
                    name="thread_name",
                    pid=pid,
                    tid=t,
                    args=dict(name=f"{rid} · {lane}"),
                )
            )
        return t

    def add(rid, lane, name, a, b=None, **args):
        t = tid(rid, lane)
        if t is None:
            return
        item = dict(name=name, pid=pid, tid=t, ts=us(a), args=args)
        if b is None:
            item.update(ph="i", s="t")
        else:
            item.update(ph="X", dur=max(b - a, 0) / 1000)
        out.append(item)

    def mine(e, rid, r):
        return owner.get((rid, r)) == e["src"]

    sends, loads, batches, sessions, prealloc = {}, {}, {}, {}, {}
    for e in run.events:
        n, rid = e["n"], run.rid_of(e)
        if n == "client.send":
            sends[e["r"]] = e
        elif n == "client.sse" and f_(e, "content") and e["r"] in sends:
            if not sends[e["r"]].get("_first"):
                sends[e["r"]]["_first"] = True
                add(e["r"], "客户端", "首内容", e["w"])
        elif n == "client.end" and e["r"] in sends:
            a = sends.pop(e["r"])
            add(e["r"], "客户端", f_(a, "tag", e["r"]), a["w"], e["w"], ok=f_(e, "ok"))
        elif n == "sched.run_batch":
            for r in f_(e, "rids") or ():
                lane = "P 调度" if role(e) == "prefill" else "D 调度"
                if mine(e, r, role(e)):
                    add(r, lane, f"run_batch {f_(e, 'mode')}", e["w"], e["w"] + e["d"])
        elif n == "uc.save.begin":
            for r in f_(e, "rids") or ():
                if mine(e, r, "prefill"):
                    sessions.setdefault((e["src"], f_(e, "session")), []).append(
                        (r, e["w"])
                    )
        elif n in ("od.put.session_done", "uc.save.consumed"):
            for r, begin in sessions.get((e["src"], f_(e, "session")), ()):
                if n == "od.put.session_done":
                    add(r, "P PUT", "PUT session", begin, e["w"], ok=f_(e, "ok"))
                else:
                    add(r, "P PUT", "scheduler 回收", e["w"])
        elif n in ("notify.ready", "notify.send", "notify.rdma_done", "notify.fail"):
            if rid and role(e) == "prefill" and mine(e, rid, "prefill"):
                add(rid, "P READY", n, e["w"], pages=f_(e, "pages"))
        elif n in ("notify.recv", "notify.recv_invalid"):
            if rid and mine(e, rid, "decode"):
                add(
                    rid,
                    "D READY",
                    n,
                    e["w"],
                    pages=f_(e, "pages"),
                    layers=f_(e, "layers"),
                )
        elif n == "linker.load_submit" and mine(e, rid, "decode"):
            loads[(e["src"], e["r"])] = e
        elif n == "linker.load_applied" and (e["src"], e["r"]) in loads:
            a = loads.pop((e["src"], e["r"]))
            add(rid, "D load", e["r"].split("@", 1)[-1], a["w"], e["w"], ok=f_(e, "ok"))
        elif n == "od.get.enqueue":
            batches[run.batch_key(e)] = e
        elif n == "od.get.done" and run.batch_key(e) in batches:
            a = batches.pop(run.batch_key(e))
            for tag in f_(a, "tags") or ():
                r = tag.split("@", 1)[0]
                if mine(a, r, "decode"):
                    name = f"GET batch {f_(e, 'batch')}"
                    add(r, "D load", name, a["w"], e["w"], ok=f_(e, "ok"))
        elif n == "pd.prealloc" and mine(e, rid, "decode"):
            prealloc[rid] = e
        elif n in ("pd.transferred", "req.abort") and mine(e, rid, "decode"):
            if rid not in prealloc:
                continue
            a = prealloc.pop(rid)
            label = "传输完成·放行" if n == "pd.transferred" else "abort"
            add(rid, "D 传输/放行", label, a["w"], e["w"])
        elif n == "pd.conclude" and rid and mine(e, rid, role(e)):
            lane = "D 传输/放行" if role(e) == "decode" else "P READY"
            add(rid, lane, f"conclude {f_(e, 'status')}", e["w"])

    # Still open at the capture cutoff: draw to the owner's last flush, named
    # as in progress (trace only; report distributions use completed ones).
    def cutoff(e):
        f = run.files[e["src"]]
        return None if f.last_flush is None else f.last_flush + f.offset

    for (_, tag), a in loads.items():
        if cutoff(a) is not None:
            name = f"{tag.split('@', 1)[-1]} 进行中"
            add(run.rid_of(a), "D load", name, a["w"], cutoff(a))
    for (_, _, batch), a in batches.items():
        for tag in f_(a, "tags") or ():
            r = tag.split("@", 1)[0]
            if mine(a, r, "decode") and cutoff(a) is not None:
                add(r, "D load", f"GET batch {batch} 进行中", a["w"], cutoff(a))
    for rid, a in prealloc.items():
        f = run.files[a["src"]]
        if f.last_flush is not None:
            add(
                rid,
                "D 传输/放行",
                "未放行（仍在进行）",
                a["w"],
                f.last_flush + f.offset,
            )
    for rid, s in states.items():
        for a, b, name in s["segments"]:
            add(rid, "D 状态", name, a, b)
        for a, b in s["hints"]:
            add(rid, "D 提示", HINT, a, b)
    out.insert(
        0, dict(ph="M", name="process_name", pid=pid, args=dict(name="请求并行轨道"))
    )
    return out


# -------------------------------------------------------------------- CLI


def cmd_report(args):
    if args.wait:
        time.sleep(args.wait)  # let live servers write their next flush
    run = Run(args.dir)
    out = args.out or args.dir
    os.makedirs(out, exist_ok=True)
    text, summary = build_report(run)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(out, "trace.json"), "w", encoding="utf-8") as f:
        json.dump(build_trace(run), f)
    print(f"report: {os.path.join(out, 'report.md')}")
    print(
        f"timeline: {os.path.join(out, 'trace.json')} (chrome://tracing or ui.perfetto.dev)"
    )


def cmd_merge(args):
    run = Run(args.dir)
    path = args.output or os.path.join(args.dir, "trace.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_trace(run), f)
    print(path)


def _load_summary(path):
    if os.path.isdir(path):
        candidate = os.path.join(path, "summary.json")
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                return json.load(f)
        return build_report(Run(path))[1]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_rows(a, b):
    rows = []

    def add(label, x, y):
        if x is None and y is None:
            return
        delta = "-"
        if x not in (None, 0) and y is not None:
            delta = f"{(y - x) / x * 100:+.1f}%"
        rows.append([label, fmt(x), fmt(y), delta])

    for rnd in sorted(
        set(a.get("client", {})) | set(b.get("client", {})),
        key=lambda r: int(r.lstrip("r") or 0),
    ):
        ca, cb = a.get("client", {}).get(rnd, {}), b.get("client", {}).get(rnd, {})
        for metric, label in (
            ("first_content", "首内容"),
            ("first_frame", "首帧"),
            ("gap", "帧间隔"),
        ):
            for q in ("p50", "max"):
                add(
                    f"{rnd} {label} {q}",
                    ca.get(metric, {}).get(q),
                    cb.get(metric, {}).get(q),
                )
    for section, label in (
        ("decode_states", "D状态合计"),
        ("get", "GET"),
        ("put", "PUT"),
    ):
        keys = set(a.get(section, {})) | set(b.get(section, {}))
        for k in sorted(keys):
            field = "sum" if section == "decode_states" else "p50"
            add(
                f"{label} {k} {field}",
                a.get(section, {}).get(k, {}).get(field),
                b.get(section, {}).get(k, {}).get(field),
            )
    return rows


def cmd_compare(args):
    a, b = _load_summary(args.a), _load_summary(args.b)
    for name, s in ((args.a, a), (args.b, b)):
        if s.get("incomplete"):
            print(f"注意：{name} 的数据不完整，见其 report.md 第 1 节。")
    print(table(["指标（ms）", "A", "B", "B 相对 A"], compare_rows(a, b)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report", help="report.md + trace.json + summary.json")
    p.add_argument("dir")
    p.add_argument("--out", help="output directory (default: the run directory)")
    p.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="seconds to wait first so running servers flush",
    )
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("merge", help="write only the Chrome/Perfetto trace")
    p.add_argument("dir")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_merge)
    p = sub.add_parser("compare", help="compare two runs (dirs or summary.json)")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
