# SPDX-License-Identifier: Apache-2.0
"""Background OceanDisk reads for PD decode.

A request's prefix is read as soon as decode has allocated its slots, while the
request still queues on prefill, so the large GET does not coincide with the
decode of the previous request. One thread per rank owns the GET stream and
never takes a scheduler turn; the scheduler only publishes whether any DP group
is decoding.

EXISTS only finds pages committed by earlier rounds. A page prefill writes for
this request counts only after its READY arrives, and a page read earlier is
read again, because EXISTS can see an object whose layers are still in flight.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# A finished forward may still run after the scheduler stops seeing decode.
_DECODE_END_HOLD_S = 0.03
# A fully loaded request reaches decode within a few scheduler iterations.
_DECODE_START_HOLD_S = 0.03
_GATED_WAIT_S = 0.0005
_IDLE_WAIT_S = 0.05
_HOLE_RECHECK_S = 0.1


def merge_ranges(ranges: list, start: int, end: int) -> list:
    """Coalesce overlapping or adjacent page ranges without bridging a gap."""
    merged = []
    for left, right in sorted([*ranges, (start, end)]):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def covered_until(ranges: list, start: int) -> int:
    for left, right in ranges:
        if left <= start < right:
            return right
    return start


class DecodeLoadGate:
    """Decides when a storage operation may start.

    Without decode every operation may run. With decode, one grant becomes
    available every ``interval`` steps and stays until the loader takes it, so a
    loader busy with the previous operation cannot miss it; grants do not pile
    up, which keeps one slow rank from bursting later. A grant admits one
    operation, EXISTS or GET alike: both run native kernels on the GPU.

    Step numbers come from the DP all-gather, so every rank gets its grant at the
    same step. A rank still finishing an operation starts later: this bounds the
    I/O per decode step on each rank, it does not make DP groups start together.
    """

    def __init__(self, interval: int, clock: Callable[[], float] = time.monotonic):
        self._interval = interval
        self._clock = clock
        self._lock = threading.Lock()
        self._active = False
        self._step = 0
        self._grant = False
        self._hold_until = 0.0

    def update(self, batch: Any) -> None:
        """Scheduler thread, once per iteration, after the DP all-gather."""
        active = False
        if batch is not None:
            tokens = getattr(batch, "global_num_tokens", None)
            active = any(tokens) if tokens else batch.forward_mode.is_decode()
        if active:
            self._step += 1
            if self._step % self._interval == 0:
                with self._lock:
                    self._grant = True
        elif self._active:
            with self._lock:
                self._grant = False
            self.hold(_DECODE_END_HOLD_S)
        self._active = active

    def hold(self, seconds: float) -> None:
        self._hold_until = max(self._hold_until, self._clock() + seconds)

    def allow(self) -> bool:
        """Loader thread: True admits exactly one operation."""
        if self._clock() < self._hold_until:
            return False
        if not self._active:
            return True
        with self._lock:
            grant, self._grant = self._grant, False
        return grant


class HandoffJob:
    """One request's pages on one rank. Guarded by the loader's condition."""

    def __init__(
        self,
        loader: DecodeHandoffLoader,
        *,
        rid: str,
        room: int,
        req: Any,
        slots: np.ndarray,
        total_len: int,
        page_size: int,
        loaded_pages: int,
        is_bigram: bool = False,
    ):
        self.loader = loader
        self.rid = rid
        self.room = room
        self.req = req
        self.slots = slots
        self.total_len = total_len
        self.page_size = page_size
        self.is_bigram = is_bigram
        self.full_pages = total_len // page_size
        self.has_tail = total_len % page_size != 0
        self.first_page = min(loaded_pages, self.full_pages)
        self.loaded = self.first_page  # Pages [0, loaded) are usable.
        self.known_end = self.first_page  # EXISTS trusted up to here.
        self.page_hashes = None
        self.object_keys = None
        self.tail_key = None
        self.checked = False
        self.ready = []  # Page ranges prefill finished writing.
        self.prefill_start = None  # First page prefill writes.
        self.prefill_done = False  # Prefill will report no more pages.
        self.tail_ready = False
        self.tail_loaded = False
        self.version = 0
        self.inflight = None
        self.failed = None
        self.closed = False
        self.done = False
        self.next_hole_check = 0.0
        self.sequence = 0

    @property
    def busy(self) -> bool:
        return self.inflight is not None

    def on_ready(self, start: int, end: int, tail: bool) -> None:
        """Mooncake thread: prefill finished writing pages [start, end)."""
        with self.loader.condition:
            self.prefill_start = (
                start if self.prefill_start is None else min(self.prefill_start, start)
            )
            self.known_end = min(self.known_end, self.prefill_start)
            if start < end:
                self.ready = merge_ranges(self.ready, start, end)
            self.tail_ready = self.tail_ready or tail
            # Anything read before this report may hold a partial object.
            keep = max(start, self.first_page)
            inflight_end = self.inflight[1] if self.inflight else 0
            if keep < inflight_end:
                self.version += 1
            if keep < self.loaded:
                self.version += 1
                self.loaded = keep
                self.done = False
            self.loader.condition.notify_all()

    def prefill_finished(self) -> None:
        """Prefill concluded, so anything still missing was not its to write."""
        with self.loader.condition:
            if not self.prefill_done:
                self.prefill_done = True
                self.loader.condition.notify_all()

    def close(self) -> None:
        """Stop reading. ``busy`` stays set until a running GET has drained."""
        with self.loader.condition:
            self.closed = True
            self.loader.condition.notify_all()

    def next_action(self, now: float, slice_pages: int) -> Optional[tuple]:
        if self.closed or self.failed is not None or self.inflight is not None:
            return None
        if self.object_keys is None:
            return ("prepare",)
        if not self.checked:
            if self.loaded < self.full_pages:
                return ("exists", self.loaded, self.full_pages)
            self.checked = True
        end = min(
            max(self.known_end, covered_until(self.ready, self.loaded)),
            self.full_pages,
        )
        if end > self.loaded:
            return ("get", self.loaded, min(end, self.loaded + slice_pages))
        if self.loaded < self.full_pages:
            # Pages neither prefill nor the first EXISTS covered, such as an
            # older PUT that finished after that query.
            hole_end = next(
                (left for left, _ in self.ready if left > self.loaded),
                self.full_pages if self.prefill_done else None,
            )
            if hole_end is not None and now >= self.next_hole_check:
                return ("exists", self.loaded, hole_end)
            return None
        if self.has_tail and self.tail_ready and not self.tail_loaded:
            return ("tail",)
        return None

    def tail_slots(self) -> np.ndarray:
        first = int(self.slots[self.full_pages * self.page_size])
        first -= first % self.page_size
        return np.arange(first, first + self.page_size, dtype=np.int64)

    def _finished(self) -> bool:
        return (
            self.inflight is None
            and self.loaded >= self.full_pages
            and (not self.has_tail or self.tail_loaded)
        )


class DecodeHandoffLoader:
    def __init__(
        self,
        backend: Any,
        gate: DecodeLoadGate,
        slice_pages: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.backend = backend
        self.gate = gate
        self.slice_pages = slice_pages
        self.clock = clock
        self.condition = threading.Condition()
        self.jobs: list[HandoffJob] = []
        self._sequence = 0
        self._closing = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="oceandisk-pd-decode-load"
        )

    def start(self) -> None:
        self._thread.start()

    def new_job(self, **fields) -> HandoffJob:
        return HandoffJob(self, **fields)

    def submit(self, job: HandoffJob) -> None:
        with self.condition:
            self._sequence += 1
            job.sequence = self._sequence
            job.done = job._finished()
            self.jobs.append(job)
            self.condition.notify_all()

    def drain(self) -> None:
        """Drop every job after any running GET has finished."""
        with self.condition:
            for job in self.jobs:
                job.closed = True
            self.condition.notify_all()
            while any(job.busy for job in self.jobs):
                self.condition.wait(_IDLE_WAIT_S)
            self.jobs.clear()

    def stop(self) -> None:
        self.drain()
        with self.condition:
            self._closing = True
            self.condition.notify_all()
        if self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        with self.backend.loader_context():
            while self._run_once():
                pass

    def _run_once(self) -> bool:
        with self.condition:
            if self._closing:
                return False
            work, wait = self._next_work()
            if work is None:
                self.condition.wait(wait)
                return True
        work()
        return True

    def _next_work(self) -> tuple:
        now = self.clock()
        self.jobs = [job for job in self.jobs if not job.closed or job.busy]
        # Requests prefill already reported are the next to reach decode.
        ordered = sorted(
            self.jobs, key=lambda job: (job.prefill_start is None, job.sequence)
        )
        pending = False
        for job in ordered:
            action = job.next_action(now, self.slice_pages)
            if action is None:
                continue
            if action[0] == "prepare":
                return lambda job=job: self._prepare(job), 0
            if not self.gate.allow():
                pending = True
                break
            if action[0] == "exists":
                _, first, last = action
                return lambda job=job, a=first, b=last: self._exists(job, a, b), 0
            if action[0] == "tail":
                job.inflight = (job.full_pages, job.full_pages, job.version)
                return lambda job=job: self._get_tail(job), 0
            _, first, last = action
            job.inflight = (first, last, job.version)
            return lambda job=job, a=first, b=last: self._get(job, a, b), 0
        return None, _GATED_WAIT_S if pending else _IDLE_WAIT_S

    def _fail(self, job: HandoffJob, error: BaseException) -> None:
        logger.exception("OceanDisk decode load failed for %s", job.rid)
        job.failed = str(error)

    def _prepare(self, job: HandoffJob) -> None:
        try:
            keys = self.backend.handoff_keys(job)
        except BaseException as error:
            with self.condition:
                self._fail(job, error)
                return
        with self.condition:
            job.page_hashes, job.object_keys, job.tail_key = keys

    def _exists(self, job: HandoffJob, first: int, last: int) -> None:
        try:
            hits = self.backend.handoff_exists(job.object_keys[first:last])
        except BaseException as error:
            with self.condition:
                self._fail(job, error)
                return
        with self.condition:
            end = first + hits
            # A later check targets pages prefill did not write.
            if not job.checked and job.prefill_start is not None:
                end = min(end, job.prefill_start)
            job.checked = True
            job.next_hole_check = self.clock() + _HOLE_RECHECK_S
            job.known_end = max(job.known_end, end)

    def _get(self, job: HandoffJob, first: int, last: int) -> None:
        page = job.page_size
        self._read(
            job,
            first,
            job.page_hashes[first:last],
            job.slots[first * page : last * page],
        )

    def _get_tail(self, job: HandoffJob) -> None:
        self._read(job, job.full_pages, [job.tail_key], job.tail_slots())

    def _read(self, job: HandoffJob, first: int, page_keys, slots) -> None:
        tag = f"{job.rid}@{first}:0"
        error = None
        try:
            self.backend.handoff_get(page_keys, slots, tag=tag, first_page=first)
        except BaseException as caught:
            error = caught
        with self.condition:
            start, end, version = job.inflight
            job.inflight = None
            if error is not None:
                self._fail(job, error)
            elif start == end:
                job.tail_loaded = True
            elif version == job.version:
                job.loaded = max(job.loaded, end)
            finished = error is None and job._finished()
            if finished and not job.done:
                job.done = True
                self.gate.hold(_DECODE_START_HOLD_S)
            self.backend.handoff_applied(
                tag, ok=error is None, loaded_pages=job.loaded, page_size=job.page_size
            )
            self.condition.notify_all()
