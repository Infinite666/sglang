"""Unit tests for the UCLinker trace recorder and its offline report tool."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
RECORDER = ROOT / "python/sglang/srt/observability/ucl_trace.py"
TOOL = ROOT / "examples/runtime/engine/Round/ucl_trace_tool.py"


def load(path, name, env):
    with patch.dict(os.environ, env, clear=False):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


tool = load(TOOL, "ucl_trace_tool", {})


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


MS = 1_000_000


def write_trace(directory, name, role, events, host="h1", ended=True, extra=""):
    """Write one synthetic process file; event times are in milliseconds."""
    lines = [
        dict(type="header", host=host, pid=name, perf_anchor_ns=0, wall_anchor_ns=0)
    ]
    lines.append(dict(t=0, n="process.identity", th=1, f=dict(role=role, label=name)))
    for item in events:
        t, n, r, f = item[:4]
        record = dict(t=int(t * MS), n=n, th=item[5] if len(item) > 5 else 1)
        if r is not None:
            record["r"] = r
        if f:
            record["f"] = f
        if len(item) > 4 and item[4] is not None:
            record["d"] = int(item[4] * MS)
        lines.append(record)
    lines.append(dict(type="flush", t=int(300 * MS), events=len(events), dropped=0))
    if ended:
        lines.append(dict(type="end", t=int(300 * MS), dropped=0))
    with open(os.path.join(directory, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x) for x in lines) + "\n" + extra)


RID = "r1-0-abcd"
ROOM = 7
GROUP_READY = "层组READY：P 层组完成被观察→notify.ready（同线程）"
RECEIPT_READY = "receipt READY：PUT session完成→notify.ready（同线程回调）"


def notify_run(directory, *, merged_wait=False):
    """One request: 2 windows x 2 layer groups of 50 layers; page=64."""
    plan = dict(
        room=ROOM,
        prefix_len=2 * 16 * 64,
        loaded_len=0,
        page=64,
        window_pages=16,
        group=50,
    )
    g0, g1 = [0, 50], [50, 61]
    d = [
        (0, "req.arrive", RID, dict(room=ROOM, input_len=3000)),
        (0, "pd.prealloc", RID, dict(room=ROOM, decode_prefix_len=2048)),
        (0, "linker.plan", RID, plan),
        (10, "notify.recv", None, dict(room=ROOM, pages=[0, 32], layers=g0)),
    ]
    if merged_wait:
        # Local rank is ready at 10 ms but the consensus allows it at 15 ms.
        d.append(
            (
                10,
                "linker.decide",
                RID,
                dict(local=1, merged=0, loaded_len=0, loaded_layers=0),
            )
        )
        d.append(
            (
                15,
                "linker.decide",
                RID,
                dict(local=1, merged=1, loaded_len=0, loaded_layers=0),
            )
        )
    d += [
        (16, "linker.load_submit", f"{RID}@0:0", dict(pages=[0, 16], layers=g0)),
        (
            16,
            "od.get.enqueue",
            None,
            dict(batch=0, tags=[f"{RID}@0:0"], pages=[16], layers=g0),
        ),
        (18, "od.get.dequeue", None, dict(batch=0), None, 2),
        (19, "od.get.dep_ready", None, dict(batch=0), None, 2),
        (19, "od.native", None, dict(op="get", wait_us=0), 4, 2),
        (20, "notify.recv", None, dict(room=ROOM, pages=[0, 32], layers=g1)),
        (40, "od.get.layer_ready", None, dict(layer=49), None, 2),
        (41, "od.get.done", None, dict(batch=0, ok=True), None, 2),
        (45, "uc.load.consumed", f"{RID}@0:0", dict(batch=0, ok=True)),
        (
            46,
            "linker.load_applied",
            f"{RID}@0:0",
            dict(ok=True, loaded_len=0, loaded_layers=50),
        ),
        (50, "linker.load_submit", f"{RID}@0:50", dict(pages=[0, 16], layers=g1)),
        (
            70,
            "linker.load_applied",
            f"{RID}@0:50",
            dict(ok=True, loaded_len=1024, loaded_layers=0),
        ),
        (72, "linker.load_submit", f"{RID}@16:0", dict(pages=[16, 32], layers=g0)),
        (
            90,
            "linker.load_applied",
            f"{RID}@16:0",
            dict(ok=True, loaded_len=1024, loaded_layers=50),
        ),
        (92, "linker.load_submit", f"{RID}@16:50", dict(pages=[16, 32], layers=g1)),
        (
            110,
            "linker.load_applied",
            f"{RID}@16:50",
            dict(ok=True, loaded_len=2048, loaded_layers=0),
        ),
        (115, "linker.kv_poll", RID, dict(state="Success")),
        (120, "pd.transferred", RID, dict(room=ROOM)),
        (121, "sched.run_batch", None, dict(mode="PREBUILT", bs=1, rids=[RID]), 0.5),
        (122, "sched.run_batch", None, dict(mode="DECODE", bs=1, rids=[RID]), 0.5),
    ]
    p = [
        (-50, "req.arrive", RID, dict(room=ROOM, input_len=3000)),
        (0.5, "uc.save.begin", None, dict(session=1, phase="prefill", rids=[RID])),
        (1, "od.put.enqueue", None, dict(session=1, layer=0)),
        (2, "od.put.dequeue", None, dict(session=1, layer=0), None, 3),
        (3, "od.put.dep_ready", None, dict(layer=0), None, 3),
        (3, "od.native", None, dict(op="put", wait_us=0), 4, 3),
        (
            8,
            "od.put.group_done",
            None,
            dict(session=1, layer_start=0, layer_end=50),
            None,
            3,
        ),
        (8.5, "od.put.session_done", None, dict(session=1, ok=True), None, 3),
        (9, "uc.save.consumed", None, dict(session=1, ok=True)),
        (
            9,
            "notify.ready",
            None,
            dict(room=ROOM, pages=[0, 32], layers=g0, origin="group"),
            None,
            3,
        ),
        (9.5, "notify.send", None, dict(room=ROOM, pages=[0, 32], layers=g0)),
        (19, "notify.send", None, dict(room=ROOM, pages=[0, 32], layers=g1)),
    ]
    c = [
        (-60, "client.send", RID, dict(tag="r1-0")),
        (125, "client.sse", RID, dict(content=False)),
        (130, "client.sse", RID, dict(content=True)),
        (140, "client.sse", RID, dict(content=True)),
        (141, "client.end", RID, dict(ok=True)),
    ]
    write_trace(directory, "decode", "decode", d)
    write_trace(directory, "prefill", "prefill", p)
    write_trace(directory, "client", "client", c)


class TestRecorder(unittest.TestCase):
    def test_disabled_without_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"SGLANG_UCL_TRACE": "1", "SGLANG_UCL_TRACE_DIR": ""}
            module = load(RECORDER, "ucl_off", env)
            self.assertFalse(module.ON)
            module = load(
                RECORDER,
                "ucl_off2",
                {"SGLANG_UCL_TRACE": "0", "SGLANG_UCL_TRACE_DIR": tmp},
            )
            self.assertFalse(module.ON)
            self.assertEqual(os.listdir(tmp), [])

    def test_records_events_stats_and_end_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load(
                RECORDER,
                "ucl_on",
                {
                    "SGLANG_UCL_TRACE": "1",
                    "SGLANG_UCL_TRACE_DIR": tmp,
                    "SGLANG_UCL_TRACE_SLOW_MS": "0",
                },
            )
            self.assertTrue(module.ON)
            module.set_identity("decode", "scheduler_TP0", tp_rank=0)
            module.emit("req.arrive", "rid-1", room=3)
            module.span("od.native", module.now() - 5000, op="get")
            for _ in range(3):
                module.stat("sched.process_queue", module.now(), slow_ns=None)
            module.stat("linker.consensus", module.now() - 10**7)
            module.close()
            module.emit("after.close")  # ignored, must not reopen or raise
            (name,) = os.listdir(tmp)
            text = read_text(os.path.join(tmp, name))
            records = [json.loads(x) for x in text.splitlines()]
            kinds = [r.get("type", r.get("n")) for r in records]
            self.assertEqual(kinds[0], "header")
            self.assertEqual(kinds[-1], "end")
            names = [r["n"] for r in records if "type" not in r]
            self.assertEqual(names[:3], ["process.identity", "req.arrive", "od.native"])
            self.assertIn("linker.consensus", names)  # slow call kept as a span
            stat = next(
                r
                for r in records
                if r.get("n") == "sched.process_queue" and r.get("type") == "stat"
            )
            self.assertEqual(stat["c"], 3)
            self.assertGreaterEqual(
                next(r for r in records if r.get("n") == "od.native")["d"], 5000
            )
            run = tool.Run(tmp)
            self.assertEqual(run.files[0].role, "decode")
            self.assertTrue(run.files[0].ended)

    def test_unwritable_directory_disables_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = os.path.join(tmp, "file")
            write_text(blocker, "")
            bad = os.path.join(blocker, "sub")
            env = {"SGLANG_UCL_TRACE": "1", "SGLANG_UCL_TRACE_DIR": bad}
            module = load(RECORDER, "ucl_bad_close", env)
            with contextlib.redirect_stderr(io.StringIO()):
                module.set_identity("decode")
                module.emit("req.arrive", "rid")  # caller does no I/O
                module.close()
            self.assertFalse(module.ON)
            env["SGLANG_UCL_TRACE_FLUSH_S"] = "0.02"
            module = load(RECORDER, "ucl_bad_writer", env)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                module.emit("req.arrive", "rid")
                module._writer.join(5)
            self.assertFalse(module.ON)
            self.assertIn("UCL trace disabled", err.getvalue())
            module.emit("after.disable")  # no-op
            self.assertEqual(len(module._events), 0)

    def test_gpu_span_maps_device_time_onto_the_trace_clock(self):
        class Event:
            def __init__(self, at):
                self.at = at

            def synchronize(self):
                pass

            def elapsed_time(self, end):
                return end.at - self.at

        fake = SimpleNamespace(cuda=SimpleNamespace(Stream=lambda: None, Event=None))
        with tempfile.TemporaryDirectory() as tmp:
            module = load(
                RECORDER,
                "ucl_gpu",
                {
                    "SGLANG_UCL_TRACE": "1",
                    "SGLANG_UCL_TRACE_DIR": tmp,
                    "SGLANG_UCL_TRACE_GPU": "1",
                },
            )
            self.assertTrue(module.GPU)
            records = []
            module.gpu_mark = lambda stream: Event(100.0)
            module.now = lambda: 5_000_000
            module._push = records.append
            with patch.dict("sys.modules", {"torch": fake}):
                module.gpu_span("gpu.get", Event(102.0), Event(110.0), pages=1)
            self.assertEqual(records[0][:3], (7_000_000, 8_000_000, "gpu.get"))

    def test_gpu_streams_records_ids_and_skips_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load(
                RECORDER,
                "ucl_streams",
                {"SGLANG_UCL_TRACE": "1", "SGLANG_UCL_TRACE_DIR": tmp},
            )
            records = []
            module._push = records.append
            stream = SimpleNamespace(stream_id=3, cuda_stream=255)
            with patch.dict(os.environ, {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}):
                module.gpu_streams("test", get=stream, copy=None)
            fields = records[0][5]
            self.assertEqual(fields["streams"], {"get": [3, "0xff"]})
            self.assertEqual(fields["max_connections"], "32")

    def test_no_file_without_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"SGLANG_UCL_TRACE": "1", "SGLANG_UCL_TRACE_DIR": tmp}
            load(RECORDER, "ucl_idle", env).close()
            self.assertEqual(os.listdir(tmp), [])

    def test_round_client_finds_recorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"SGLANG_UCL_TRACE": "1", "SGLANG_UCL_TRACE_DIR": tmp}
            rounds = load(
                ROOT / "examples/runtime/engine/Round/run_rounds.py",
                "rounds_traced",
                env,
            )
            self.assertIsNotNone(rounds.ucl_trace)
            self.assertTrue(rounds.UCL_CLIENT_RID)

    def test_bounded_buffer_counts_drops(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = load(
                RECORDER,
                "ucl_small",
                {
                    "SGLANG_UCL_TRACE": "1",
                    "SGLANG_UCL_TRACE_DIR": tmp,
                    "SGLANG_UCL_TRACE_MAX_BUFFERED": "2",
                    "SGLANG_UCL_TRACE_FLUSH_S": "60",
                },
            )
            for i in range(5):
                module.emit("e", str(i))
            module.close()
            run = tool.Run(tmp)
            self.assertEqual(run.files[0].dropped, 3)
            _, problems, _ = tool.completeness(run)
            self.assertTrue(any("丢弃" in p for p in problems))


class TestReport(unittest.TestCase):
    def test_states_hint_and_storage_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            notify_run(tmp)
            run = tool.Run(tmp)
            states = tool.decode_states(run)
            acc = {}
            for a, b, name in states[RID]["segments"]:
                acc[name] = acc.get(name, 0) + (b - a) / MS
            self.assertAlmostEqual(acc["等待READY"], 10)
            self.assertAlmostEqual(acc["READY已到·未提交"], 6 + 4 + 2 + 2)
            self.assertAlmostEqual(acc["load进行中"], 30 + 20 + 18 + 18)
            self.assertAlmostEqual(acc["prefix完成·等传输/放行"], 10)
            self.assertAlmostEqual(sum(acc.values()), 120)
            # The next window/layer group was READY while the current load ran:
            # 20-46 (group 1), 50-70 (window 1 group 0), 72-90 (window 1 group 1).
            hint = sum((b - a) / MS for a, b in states[RID]["hints"])
            self.assertAlmostEqual(hint, 26 + 20 + 18)
            get, put, native, _, _ = tool.storage_section(run)
            self.assertEqual(get["入队→worker取出"], [2.0])
            self.assertEqual(get["取出→前置依赖就绪"], [1.0])
            self.assertEqual(get["最后native返回→worker观察完成"], [18.0])
            self.assertEqual(get["worker观察完成→scheduler消费"], [4.0])
            self.assertEqual(native["get"], [4.0])
            # A real PUT chain (it used to crash the report with KeyError).
            self.assertEqual(native["put"], [4.0])
            self.assertEqual(put["入队→worker取出(每层)"], [1.0])
            self.assertEqual(put["取出→前置依赖就绪(每层)"], [1.0])
            self.assertEqual(put["层组最后native返回→worker观察完成"], [1.0])
            self.assertEqual(put["session开始→worker观察完成"], [8.0])
            self.assertEqual(put["session完成→scheduler回收"], [0.5])
            delays, ready, ambiguous, unmatched = tool.notify_section(run)
            self.assertEqual(delays, [0.5, 1.0])
            self.assertEqual(ready[GROUP_READY], [1.0])
            self.assertEqual((ambiguous, unmatched), (0, 0))
            self.assertEqual(states[RID]["outcome"], "released")
            self.assertEqual(states[RID]["open_ms"], 0)

    def test_rank_wait_is_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            notify_run(tmp, merged_wait=True)
            states = tool.decode_states(tool.Run(tmp))
            acc = {}
            for a, b, name in states[RID]["segments"]:
                acc[name] = acc.get(name, 0) + (b - a) / MS
            self.assertAlmostEqual(acc["本rank就绪·等其他rank"], 5)

    def test_all_layer_range_shorter_than_window_counts_as_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = dict(room=1, prefix_len=4 * 64, page=64, window_pages=2, group=50)
            rng = dict(room=1, pages=[0, 1])
            write_trace(
                tmp,
                "decode",
                "decode",
                [
                    (0, "linker.plan", "r1-1-x", plan),
                    (10, "notify.recv", None, dict(rng, layers=[0, 50])),
                    (10, "notify.recv", None, dict(rng, layers=[50, 61])),
                    (
                        10,
                        "linker.decide",
                        "r1-1-x",
                        dict(local=1, merged=1, ends=[1, 1]),
                    ),
                    (
                        20,
                        "linker.load_submit",
                        "r1-1-x@0:0",
                        dict(pages=[0, 1], layers=[0, 61]),
                    ),
                    (30, "pd.transferred", "r1-1-x", dict(room=1)),
                ],
            )
            acc = {}
            for a, b, name in tool.decode_states(tool.Run(tmp))["r1-1-x"]["segments"]:
                acc[name] = acc.get(name, 0) + (b - a) / MS
            self.assertEqual(acc["等待READY"], 10)
            self.assertEqual(acc["READY已到·未提交"], 10)

    def test_window_stops_unfinished_states_at_window_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = dict(room=1, prefix_len=1024, page=64, window_pages=16, group=50)
            write_trace(
                tmp,
                "client",
                "client",
                [
                    (10, "client.send", "r1-1-x", dict(tag="r1-1")),
                    (30, "client.end", "r1-1-x", dict(ok=False)),
                ],
            )
            write_trace(
                tmp,
                "decode",
                "decode",
                [
                    (12, "linker.plan", "r1-1-x", plan),
                    (
                        15,
                        "linker.load_submit",
                        "r1-1-x@0:0",
                        dict(pages=[0, 16], layers=[0, 50]),
                    ),
                    (100, "pd.transferred", "r1-1-x", dict(room=1)),
                ],
            )
            run = tool.Run(tmp)
            s = tool.decode_states(run.windowed(10 * MS, 30 * MS))["r1-1-x"]
            self.assertTrue(s["window_cut"])
            self.assertAlmostEqual(s["open_ms"], 15)
            self.assertAlmostEqual(s["segments"][-1][1] / MS, 30)
            self.assertIn(
                "D 窗口结束时未放行（停在 load进行中 15 ms，全程采集中随后放行）",
                tool.build_report(run)[0],
            )
            # The unwindowed capture still sees the release.
            self.assertEqual(tool.decode_states(run)["r1-1-x"]["outcome"], "released")

    def test_report_files_client_columns_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            notify_run(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                tool.main(["report", tmp])
            text = read_text(os.path.join(tmp, "report.md"))
            self.assertIn("首内容 = 第一个带 content 的帧", text)
            self.assertIn("| r1 | 1/1 | 185.0 | 190.0 |", text)
            trace = json.loads(read_text(os.path.join(tmp, "trace.json")))
            names = {e["args"]["name"] for e in trace["traceEvents"] if e["ph"] == "M"}
            self.assertIn("请求并行轨道", names)
            for lane in (
                "客户端",
                "P PUT",
                "P READY",
                "D READY",
                "D 状态",
                "D load",
                "D 调度",
            ):
                self.assertIn(f"{RID} · {lane}", names)
            spans = {e["name"] for e in trace["traceEvents"] if e["ph"] == "X"}
            for name in (
                "GET batch 0",
                "PUT session",
                "0:50",
                "传输完成·放行",
                "run_batch DECODE",
            ):
                self.assertIn(name, spans)
            summary = json.loads(read_text(os.path.join(tmp, "summary.json")))
            self.assertFalse(summary["incomplete"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                tool.main(["compare", tmp, tmp])
            self.assertIn("r1 首内容 p50 | 190.0 | 190.0 | +0.0%", out.getvalue())

    def test_killed_process_and_unreleased_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = dict(
                room=1,
                prefix_len=1024,
                loaded_len=0,
                page=64,
                window_pages=16,
                group=50,
            )
            write_trace(
                tmp,
                "decode",
                "decode",
                [
                    (0, "linker.plan", "r1-1-x", plan),
                    (
                        5,
                        "linker.load_submit",
                        "r1-1-x@0:0",
                        dict(pages=[0, 16], layers=[0, 50]),
                    ),
                    (
                        5,
                        "od.get.enqueue",
                        None,
                        dict(batch=3, tags=["r1-1-x@0:0"], pages=[16], layers=[0, 50]),
                    ),
                ],
                ended=False,
                extra='{"t": 9, "n": "torn',
            )
            run = tool.Run(tmp)
            rows, problems, unclosed = tool.completeness(run)
            self.assertIn("损坏行 1", rows[0][-1])
            self.assertTrue(any("没有结束标记" in p for p in problems))
            self.assertEqual(
                {r[0] for r in unclosed}, {"od.get.enqueue", "linker.load_submit"}
            )
            text, summary = tool.build_report(run)
            self.assertTrue(summary["incomplete"])
            # The stuck load and GET batch are drawn on the request's D load
            # lane up to the last flush.
            trace = tool.build_trace(run)["traceEvents"]
            open_ops = {
                e["name"]: e["ts"] + e["dur"]
                for e in trace
                if e.get("ph") == "X" and e["name"].endswith(" 进行中")
            }
            self.assertEqual(
                open_ops,
                {"0:0 进行中": 300_000, "GET batch 3 进行中": 300_000},
            )
            # The stall after the last event is counted up to the last flush.
            s = tool.decode_states(run)["r1-1-x"]
            self.assertEqual(s["outcome"], "open")
            self.assertAlmostEqual(s["open_ms"], 295)
            self.assertEqual(s["segments"][-1][2], "load进行中")
            self.assertAlmostEqual(s["segments"][-1][1] / MS, 300)
            self.assertIn(
                "| r1-1-x | D 未放行（停在 load进行中 295 ms，仍在进行） "
                "| linker.load_submit |",
                text,
            )

    def test_abort_is_not_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = dict(room=2, prefix_len=1024, page=64, window_pages=16, group=50)
            write_trace(
                tmp,
                "decode",
                "decode",
                [
                    (0, "linker.plan", "r1-2-x", plan),
                    (7, "req.abort", "r1-2-x", dict(reason="boom")),
                    (9, "pd.transferred", "r1-2-x", dict(room=2)),
                ],
            )
            run = tool.Run(tmp)
            s = tool.decode_states(run)["r1-2-x"]
            self.assertEqual(s["outcome"], "aborted")
            self.assertAlmostEqual(sum((b - a) / MS for a, b, _ in s["segments"]), 7)
            text = tool.build_report(run)[0]
            self.assertIn("| abort |", text)
            self.assertNotIn("已放行", text)

    def test_notify_pairs_per_receiver_and_skips_ambiguous_senders(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = dict(room=1, pages=[0, 8], layers=[0, 50])
            write_trace(
                tmp,
                "p0",
                "prefill",
                [
                    (1, "notify.send", None, dict(rng, dst="d0")),
                    (2, "notify.send", None, dict(rng, dst="d1")),
                    (3, "notify.send", None, dict(rng, dst="dx")),
                ],
            )
            write_trace(
                tmp, "p1", "prefill", [(4, "notify.send", None, dict(rng, dst="dx"))]
            )
            write_trace(
                tmp,
                "d",
                "decode",
                [
                    (5, "notify.recv", None, dict(rng, dst="d1")),
                    (6, "notify.recv", None, dict(rng, dst="d0")),
                    (7, "notify.recv", None, dict(rng, dst="dx")),
                ],
            )
            delays, _, ambiguous, unmatched = tool.notify_section(tool.Run(tmp))
            self.assertEqual(sorted(delays), [3.0, 5.0])
            self.assertEqual((ambiguous, unmatched), (1, 0))

    def test_notify_receive_without_earlier_send_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = dict(room=1, dst="d0", pages=[0, 8], layers=[0, 50])
            write_trace(tmp, "p", "prefill", [(5, "notify.send", None, rng)])
            write_trace(tmp, "d", "decode", [(4, "notify.recv", None, rng)])
            run = tool.Run(tmp)
            self.assertEqual(tool.notify_section(run)[::3], ([], 1))
            self.assertIn("1 条接收找不到更早的发送记录", tool.build_report(run)[0])

    def test_cross_host_notify_delay_is_not_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            notify_run(tmp)
            path = os.path.join(tmp, "prefill.jsonl")
            write_text(path, read_text(path).replace('"host": "h1"', '"host": "h2"'))
            run = tool.Run(tmp)
            self.assertTrue(run.cross_host)
            self.assertEqual(tool.notify_section(run)[0], [])
            self.assertIn("跨机未校准", tool.build_report(run)[0])

    def test_get_batches_pair_within_reset_generation(self):
        def batch(t0, n):
            return [
                (t0, "od.get.enqueue", None, dict(batch=n, tags=[], pages=[1])),
                (t0 + 1, "od.get.dequeue", None, dict(batch=n), None, 2),
                (t0 + 2, "od.get.done", None, dict(batch=n, ok=True), None, 2),
                (t0 + 3, "uc.load.consumed", "x", dict(batch=n, ok=True)),
            ]

        for marker in ([(50, "od.reset", None, {})], []):  # explicit / fallback
            with tempfile.TemporaryDirectory() as tmp:
                events = batch(0, 0) + batch(10, 1) + marker + batch(100, 0)
                write_trace(tmp, "decode", "decode", events)
                run = tool.Run(tmp)
                get = tool.storage_section(run)[0]
                self.assertEqual(get["入队→worker取出"], [1.0, 1.0, 1.0])
                self.assertEqual(get["入队→worker观察完成"], [2.0, 2.0, 2.0])
                self.assertEqual(get["worker观察完成→scheduler消费"], [1.0] * 3)
                self.assertEqual(tool.completeness(run)[2], [])

    def test_ready_origins_and_receipt_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = dict(room=1, pages=[0, 8], layers=[0, 61])
            write_trace(
                tmp,
                "p",
                "prefill",
                [
                    (
                        1,
                        "od.put.session_done",
                        None,
                        dict(session=1, receipt=11),
                        None,
                        3,
                    ),
                    (
                        2,
                        "notify.ready",
                        None,
                        dict(rng, origin="receipt", receipt=11),
                        None,
                        3,
                    ),
                    # Registered after completion: fires in the scheduler thread.
                    (9, "notify.ready", None, dict(rng, origin="receipt", receipt=11)),
                    (10, "notify.ready", None, dict(rng, origin="exists")),
                    (11, "od.put.group_done", None, dict(layer_start=0, layer_end=50)),
                    (12, "notify.ready", None, dict(rng, layers=[50, 61])),
                ],
            )
            ready = tool.notify_section(tool.Run(tmp))[1]
            self.assertEqual(ready[RECEIPT_READY], [1.0])
            self.assertEqual(ready[GROUP_READY], [])
            self.assertEqual(ready["receipt READY（注册时已完成，不计延迟）"], 1)
            self.assertEqual(ready["存储查询命中即发 READY"], 1)
            self.assertEqual(ready["来源未知 READY（旧 trace，不计延迟）"], 1)

    def test_client_joins_server_by_sse_id_and_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_trace(
                tmp,
                "client",
                "client",
                [
                    (10, "client.send", "r1-0-x", dict(tag="r1-0", sent_rid=True)),
                    (20, "client.sse", "r1-0-x", dict(content=True, id="chatcmpl-srv")),
                    (30, "client.end", "r1-0-x", dict(ok=True)),
                ],
            )
            write_trace(
                tmp,
                "p",
                "prefill",
                [
                    (1, "sched.recv_requests", None, {}, 9),  # warm-up, outside
                    (11, "req.arrive", "srv", dict(room=1)),
                    (12, "sched.run_batch", None, dict(mode="EXTEND", rids=["srv"]), 5),
                ],
            )
            run = tool.Run(tmp)
            rows, _, _, _, server_round, via_sse = tool.client_section(run)
            self.assertEqual(rows[0][-1], "1/1")
            self.assertEqual((server_round, via_sse), ({"srv": "r1"}, 1))
            text = tool.build_report(run)[0]
            self.assertIn("按 SSE 帧的 id（服务端 rid）关联", text)
            section6, section9 = text.split("## 9.")[0], text.split("## 9.")[1]
            self.assertNotIn("sched.recv_requests", section6.split("## 6.")[1])
            self.assertIn("sched.recv_requests", section9)
            self.assertIn("| r1 | P排队 | 1 | 1.00 |", text)

    def test_prefill_supply_and_exists_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_trace(
                tmp,
                "p",
                "prefill",
                [
                    (0, "req.arrive", "a", dict(room=1)),
                    (1, "req.arrive", "b", dict(room=2)),
                    (2, "od.put.enqueue", None, dict(session=1, layer=0)),
                    (2, "sched.run_batch", None, dict(mode="EXTEND", rids=["a"]), 10),
                    (13, "od.exists", None, dict(keys=4), 2),
                    (15, "od.exists", None, dict(keys=4), 1),
                    (17, "sched.run_batch", None, dict(mode="EXTEND", rids=["b"]), 10),
                    (30, "notify.rdma_done", None, dict(room=2)),
                ],
            )
            p = tool.prefill_section(tool.Run(tmp), {})
            b = p["per_round"]["全部"]
            self.assertEqual(sorted(b["P排队"]), [2.0, 16.0])
            self.assertEqual(b["P执行跨度"], [10.0, 10.0])
            self.assertEqual(b["末次EXTEND结束→RDMA完成"], [3.0])
            self.assertEqual(p["gaps"]["有请求在等"], [5.0])
            self.assertEqual(p["gap_exists"]["有请求在等"], [3.0])
            self.assertEqual((p["exists_count"], p["admitted"]), (2, 2))

    def test_decode_intervals_split_by_explicit_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            dec = lambda t, rids: (  # noqa: E731
                t,
                "sched.run_batch",
                None,
                dict(mode="DECODE", rids=rids),
                1,
            )
            write_trace(
                tmp,
                "d",
                "decode",
                [
                    dec(0, ["a"]),
                    (5, "od.get.enqueue", None, dict(batch=0, tags=[], pages=[1])),
                    (8, "od.get.done", None, dict(batch=0, ok=True), None, 2),
                    dec(30, ["a"]),  # GET overlapped
                    (40, "linker.consensus", None, {}, 3),
                    dec(60, ["a", "b"]),  # slow consensus, new request
                    (70, "sched.run_batch", None, dict(mode="IDLE"), 1),
                    dec(500, ["a"]),  # idle in between
                    dec(530, ["c"]),  # no shared rid: undecidable, kept
                ],
            )
            kinds, groups, steps = tool.decode_step_section(tool.Run(tmp))
            self.assertEqual(len(steps), 2)
            self.assertEqual(kinds["持续decode"], [30.0, 30.0])
            self.assertEqual(kinds["含空闲"], [440.0])
            self.assertEqual(kinds["无法判定"], [30.0])
            self.assertEqual(groups["本进程GET在途"], [30.0])
            self.assertEqual(groups["其他D无GET在途"], [30.0, 30.0])
            self.assertEqual(groups["有≥2ms consensus"], [30.0])
            self.assertEqual(groups["有新请求加入"], [30.0])

    def test_p_and_d_rids_join_by_room_regardless_of_event_order(self):
        for decode_first in (False, True):
            with self.subTest(
                decode_first=decode_first
            ), tempfile.TemporaryDirectory() as tmp:
                offset = 10 if decode_first else 0
                write_trace(
                    tmp,
                    "p",
                    "prefill",
                    [
                        (offset, "req.arrive", "p-rid", dict(room=1)),
                        (
                            offset + 1,
                            "sched.run_batch",
                            None,
                            dict(mode="EXTEND", rids=["p-rid"]),
                            5,
                        ),
                        (
                            offset + 2,
                            "sched.run_batch",
                            None,
                            dict(mode="EXTEND", rids=["health"], tokens=4),
                            1,
                        ),
                        (offset + 9, "notify.rdma_done", None, dict(room=1)),
                    ],
                )
                write_trace(
                    tmp, "d", "decode", [(5, "req.arrive", "d-rid", dict(room=1))]
                )
                p = tool.prefill_section(tool.Run(tmp), {"d-rid": "r1"})
                self.assertEqual(p["per_round"]["r1"]["末次EXTEND结束→RDMA完成"], [3.0])
                self.assertNotIn("全部", p["per_round"])
                self.assertEqual(p["admitted"], 1)
                self.assertEqual([u["rid"] for u in p["unmatched"]], ["health"])
                self.assertEqual(
                    (
                        p["tail"]["完成事件"],
                        p["tail"]["关联到请求"],
                        p["tail"]["可算延迟"],
                    ),
                    (1, 1, 1),
                )

    def test_get_phases_keep_submit_apart_from_io_after_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_trace(
                tmp,
                "d",
                "decode",
                [
                    (0, "od.get.enqueue", None, dict(batch=0, tags=[], pages=[1])),
                    (1, "od.get.dequeue", None, dict(batch=0), None, 2),
                    (2, "od.get.dep_ready", None, dict(batch=0), None, 2),
                    (10, "od.native", None, dict(op="get"), 1, 2),
                    (50, "od.get.done", None, dict(batch=0, ok=True), None, 2),
                ],
            )
            batch = tool.get_phases(tool.Run(tmp))[0]
            ms = {
                label: (batch[end] - batch[start]) / 1e6
                for label, start, end in tool.GET_PHASES
            }
            self.assertEqual(
                ms, {"排队": 1, "依赖等待": 1, "构建": 8, "提交": 1, "等完成": 39}
            )

    def test_spike_merges_ranks_and_attributes_other_group_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            dec = lambda t: (  # noqa: E731
                t,
                "sched.run_batch",
                None,
                dict(mode="DECODE", rids=["a"]),
                1,
            )
            stalled = [dec(0), dec(25), dec(725), dec(750)]
            write_trace(
                tmp,
                "scheduler_DP0_TP0",
                "decode",
                [*stalled, (100, "sched.copy_done", None, dict(mode="DECODE"), 600)],
            )
            write_trace(tmp, "scheduler_DP0_TP1", "decode", stalled)
            write_trace(
                tmp,
                "scheduler_DP1_TP2",
                "decode",
                [
                    *stalled,
                    (30, "od.get.dequeue", None, dict(batch=0), None, 2),
                    (31, "od.get.dep_ready", None, dict(batch=0), None, 2),
                    (531, "od.native", None, dict(op="get"), 1, 2),
                    (600, "od.get.done", None, dict(batch=0, ok=True), None, 2),
                ],
            )
            run = tool.Run(tmp)
            spikes = tool.decode_spike_section(
                run, tool.decode_step_section(run)[2], top=1
            )
            self.assertEqual(len(spikes), 1)
            self.assertEqual(len(spikes[0]["ranks"]), 3)
            self.assertEqual(spikes[0]["ms"], 700)
            self.assertEqual(spikes[0]["top"], "其他组GET·构建")
            self.assertEqual(spikes[0]["copy"], 600)

    def test_spike_scopes_session_lifetime_and_other_rank_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            dec = lambda t: (  # noqa: E731
                t,
                "sched.run_batch",
                None,
                dict(mode="DECODE", rids=["a"]),
                1,
            )
            write_trace(tmp, "scheduler_DP0_TP0", "decode", [dec(0), dec(700)])
            write_trace(
                tmp,
                "scheduler_DP1_TP1",
                "decode",
                [
                    (0, "sched.copy_done", None, dict(mode="IDLE"), 700),
                    (0, "pd.evict", None, dict(wanted=1, freed=1), 700),
                ],
            )
            write_trace(
                tmp,
                "p",
                "prefill",
                [
                    (0, "uc.save.begin", None, dict(session=1)),
                    (0, "uc.save.begin", None, dict(session=2)),
                    (700, "od.put.session_done", None, dict(session=1)),
                    (700, "od.put.session_done", None, dict(session=2)),
                ],
            )
            run = tool.Run(tmp)
            spike = tool.decode_spike_section(run, tool.decode_step_section(run)[2])[0]
            self.assertEqual(spike["cells"][("P保存session", "")], 700)
            self.assertEqual(spike["cells"][("其他copy_done", "其他IDLE")], 700)
            self.assertEqual(spike["top"], "无")
            self.assertEqual(spike["copy"], 0)
            self.assertEqual(tool.spike_cell(spike, "驱逐"), "-")

    def test_spike_keeps_get_running_across_the_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            dec = lambda t: (  # noqa: E731
                t,
                "sched.run_batch",
                None,
                dict(mode="DECODE", rids=["a"]),
                1,
            )
            enqueue = dict(batch=0, tags=[], pages=[1])
            write_trace(
                tmp,
                "scheduler_DP0_TP0",
                "decode",
                [
                    (0, "od.get.enqueue", None, enqueue),
                    (1, "od.get.dequeue", None, dict(batch=0), None, 2),
                    (2, "od.get.dep_ready", None, dict(batch=0), None, 2),
                    (3, "od.native", None, dict(op="get"), 1, 2),
                    (5, "od.get.enqueue", None, dict(batch=1, tags=[], pages=[2])),
                    dec(10),
                    dec(90),
                    (100, "od.get.done", None, dict(batch=0, ok=True), None, 2),
                ],
            )
            run = tool.Run(tmp)
            view = run.windowed(10 * MS, 90 * MS)
            _, groups, steps = tool.decode_step_section(view, run)
            self.assertEqual(groups["本进程GET在途"], [80])
            spike = tool.decode_spike_section(run, steps)[0]
            self.assertEqual(spike["cells"][("本进程GET", "等完成")], 80)
            self.assertEqual(spike["cells"][("本进程GET", "排队·截尾")], 80)

    def test_spike_shows_gpu_time_and_keeps_p_get_out_of_the_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            dec = lambda t: (  # noqa: E731
                t,
                "sched.run_batch",
                None,
                dict(mode="DECODE", rids=["a"]),
                1,
            )
            write_trace(
                tmp,
                "scheduler_DP0_TP0",
                "decode",
                [
                    dec(0),
                    dec(700),
                    (0, "gpu.forward", None, dict(mode="DECODE"), 25),
                    (10, "gpu.get", None, None, 600),
                ],
            )
            write_trace(
                tmp,
                "p",
                "prefill",
                [
                    (0, "od.get.dequeue", None, dict(batch=0), None, 2),
                    (1, "od.get.dep_ready", None, dict(batch=0), None, 2),
                    (2, "od.native", None, dict(op="get"), 1, 2),
                    (700, "od.get.done", None, dict(batch=0, ok=True), None, 2),
                ],
            )
            run = tool.Run(tmp)
            spike = tool.decode_spike_section(run, tool.decode_step_section(run)[2])[0]
            self.assertEqual(spike["top"], "无")
            self.assertEqual(spike["cells"][("GPU forward", "")], 25)
            self.assertEqual(spike["cells"][("GPU GET", "本进程")], 600)
            self.assertEqual(spike["cells"][("P", "等完成")], 697)

    def test_decode_groups_split_p_get_when_d_has_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            dec = lambda t: (  # noqa: E731
                t,
                "sched.run_batch",
                None,
                dict(mode="DECODE", rids=["a"]),
                1,
            )
            write_trace(tmp, "d", "decode", [dec(0), dec(25), dec(50)])
            write_trace(
                tmp,
                "p",
                "prefill",
                [
                    (0, "od.get.enqueue", None, dict(batch=0, tags=[], pages=[1])),
                    (20, "od.get.done", None, dict(batch=0, ok=True)),
                ],
            )
            run = tool.Run(tmp)
            groups = tool.decode_step_section(run)[1]
            self.assertEqual(groups["D无GET·P有GET在途"], [25])
            self.assertEqual(groups["D无GET·P无GET"], [25])

    def test_room_zero_is_not_a_shared_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_trace(
                tmp,
                "p",
                "prefill",
                [
                    (0, "req.arrive", "HEALTH_CHECK_a", dict(room=0)),
                    (1, "req.arrive", "HEALTH_CHECK_b", dict(room=0)),
                    (
                        2,
                        "sched.run_batch",
                        None,
                        dict(mode="EXTEND", rids=["HEALTH_CHECK_a"]),
                        5,
                    ),
                ],
            )
            p = tool.prefill_section(tool.Run(tmp), {"client": "r1"})
            self.assertFalse(p["conflicts"])
            self.assertEqual([u["rid"] for u in p["unmatched"]], ["HEALTH_CHECK_a"])

    def test_stream_section_reports_aliased_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = dict(
                owner="scheduler",
                max_connections="8",
                streams={"forward": [1, "0xa"], "copy": [2, "0xb"]},
            )
            storage = dict(
                owner="oceandisk",
                max_connections="8",
                streams={"get": [1, "0xa"], "put": [3, "0xc"]},
            )
            write_trace(
                tmp,
                "d",
                "decode",
                [
                    (0, "gpu.streams", None, storage),
                    (1, "gpu.streams", None, scheduler),
                ],
            )
            rows, aliased = tool.stream_section(tool.Run(tmp))
            self.assertEqual(rows, [["d", "8", 1, "-", 2, 1, 3, "forward=get"]])
            self.assertEqual(len(aliased), 1)

    def test_get_spans_mark_missing_boundaries_unknown(self):
        batch = dict(src=0, queue=0, wait=1, end=9)
        self.assertEqual(tool.get_spans(batch), [("排队", 0, 1), ("未知", 1, 9)])

    def test_room_with_two_identities_is_listed_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_trace(
                tmp,
                "p",
                "prefill",
                [
                    (0, "req.arrive", "p-rid", dict(room=1)),
                    (
                        1,
                        "sched.run_batch",
                        None,
                        dict(mode="EXTEND", rids=["p-rid"]),
                        5,
                    ),
                ],
            )
            write_trace(
                tmp,
                "d",
                "decode",
                [
                    (2, "req.arrive", "d1", dict(room=1)),
                    (3, "req.arrive", "d2", dict(room=1)),
                ],
            )
            p = tool.prefill_section(tool.Run(tmp), {"d1": "r1", "d2": "r2"})
            self.assertFalse(p["per_round"])
            self.assertEqual(p["admitted"], 0)
            self.assertEqual(p["conflicts"][0]["decode"], ["d1", "d2"])
            self.assertEqual(p["conflicts"][0]["rounds"], ["r1", "r2"])

    def test_decode_cache_keeps_logical_pages_apart_from_get_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = [
                (
                    1,
                    "pd.prealloc",
                    "a",
                    dict(room=1, local_prefix=64, decode_prefix_len=640),
                ),
                (2, "linker.load_submit", "a@1:0", dict(pages=[1, 5], layers=[0, 50])),
                (
                    3,
                    "linker.load_submit",
                    "a@1:50",
                    dict(pages=[1, 5], layers=[50, 61]),
                ),
                (4, "pd.evict", "a", dict(wanted=8, freed=16), 2),
            ]
            write_trace(tmp, "scheduler_DP0_TP0", "decode", events)
            write_trace(tmp, "scheduler_DP0_TP1", "decode", events)
            run = tool.Run(tmp)
            rounds, evicted = tool.decode_cache_section(run, {"a": "r1"})
            self.assertEqual(rounds["r1"]["hit"], [64])
            self.assertEqual(rounds["r1"]["need"], [4])
            self.assertEqual(rounds["r1"]["work"], [2 * 4 * 61])
            self.assertEqual(evicted[("轮次外", "0")], [1, 16, 2.0])


if __name__ == "__main__":
    unittest.main()
