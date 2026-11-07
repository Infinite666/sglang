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

Prefill reports a page range once per layer group, as that group's PUT lands,
so decode reads the first groups while the last ones are still being written.
Neighbouring groups that can read the same pages share one GET.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable, Optional, Sequence

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


def group_span(
    groups: Sequence[tuple[int, int]], first_layer: int, end_layer: int
) -> Optional[tuple[int, int]]:
    """Groups [lo, hi) that make up layers [first_layer, end_layer), or None."""
    starts = [start for start, _ in groups]
    ends = [end for _, end in groups]
    if first_layer not in starts or end_layer not in ends:
        return None
    lo, hi = starts.index(first_layer), ends.index(end_layer) + 1
    return (lo, hi) if lo < hi else None


class DecodeLoadGate:
    """Decides when a storage operation may start.

    Without decode every operation may run. With decode, the boundary every
    ``interval`` steps issues one grant, for EXISTS or GET alike: both run
    native kernels on the GPU. A grant admits only work that was already due
    when its boundary passed. Work that becomes due later, such as pages
    prefill reports while an earlier GET runs, waits for the next boundary.
    Work already due when a boundary passes during a running operation starts
    once that operation ends, so a long operation does not make it miss the
    grant. An idle loader is issued none, and grants do not pile up, which
    keeps one slow rank from bursting later.

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
        self._epoch = 0  # Boundaries passed: work is stamped with it when due.
        self._grant = None  # Epoch of the boundary that issued the unused grant.
        self._hold_until = 0.0
        # Loader state: a boundary issues a grant only to work that wants one.
        self._running = False
        self._waiting = False

    @property
    def epoch(self) -> int:
        return self._epoch

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
                    self._epoch += 1
                    if self._running or self._waiting:
                        self._grant = self._epoch
        elif self._active:
            with self._lock:
                self._grant = None
            self.hold(_DECODE_END_HOLD_S)
        self._active = active

    def hold(self, seconds: float) -> None:
        self._hold_until = max(self._hold_until, self._clock() + seconds)

    def limit(self) -> Optional[float]:
        """Loader thread: work due before this epoch may start; None admits none."""
        with self._lock:
            if self._clock() < self._hold_until:
                return None
            if not self._active:
                return math.inf
            return self._grant

    def start(self) -> None:
        """Loader thread: an admitted operation starts and uses up the grant."""
        with self._lock:
            self._grant = None
            self._running = True
            self._waiting = False

    def release(self) -> None:
        """Loader thread: the admitted operation has ended."""
        with self._lock:
            self._running = False

    def defer(self, limit: Optional[float]) -> None:
        """Loader thread: due work cannot start under ``limit``.

        The next boundary grants it. A grant no due work may use is dropped,
        since work that becomes due from now on is newer still.
        """
        with self._lock:
            self._waiting = True
            if limit is not None and self._grant == limit:
                self._grant = None

    def idle(self) -> None:
        """Loader thread: no operation is due, so an unused grant is dropped."""
        with self._lock:
            self._waiting = False
            self._grant = None


class HandoffJob:
    """One request's pages on one rank. Guarded by the loader's condition.

    Prefill reports a page range once per layer group, so every group keeps its
    own read cursor, reported ranges and version. EXISTS covers all of them.
    Each report and EXISTS result also keeps the gate epoch it arrived in, so a
    grant admits only the pages that were readable before its boundary.
    """

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
        self.groups = loader.groups  # Layer ranges, in order.
        count = len(self.groups)
        self.full_pages = total_len // page_size
        self.has_tail = total_len % page_size != 0
        self.first_page = min(loaded_pages, self.full_pages)
        # Per group: pages [0, n) are usable on its layers.
        self.group_loaded = [self.first_page] * count
        self.known_end = self.first_page  # EXISTS trusted up to here.
        self.known_log = []  # (end, epoch) of every EXISTS that moved known_end.
        self.page_hashes = None
        self.object_keys = None
        self.tail_key = None
        self.checked = False
        self.ready = [[] for _ in range(count)]  # Per group: pages prefill wrote.
        self.ready_log = [[] for _ in range(count)]  # (start, end, epoch) of each.
        self.prefill_start = None  # First page prefill writes.
        self.prefill_done = False  # Prefill will report no more pages.
        self.tail_ready = [False] * count
        self.tail_loaded = False
        self.due_epoch = 0  # The first EXISTS is due from submission.
        self.hole_epoch = None  # A hole re-check is due from here.
        self.tail_epoch = None  # The tail is due from here.
        self.versions = [0] * count
        # A running GET: (first group, end group, first page, end page, versions).
        self.inflight = None
        self.failed = None
        self.closed = False
        self.done = False
        self.next_hole_check = 0.0
        self.sequence = 0

    @property
    def busy(self) -> bool:
        return self.inflight is not None

    @property
    def loaded(self) -> int:
        """Pages [0, loaded) are usable on every layer."""
        return min(self.group_loaded)

    def layer_span(self, first_layer: int, end_layer: int) -> Optional[tuple]:
        return group_span(self.groups, first_layer, end_layer)

    def on_ready(self, start: int, end: int, tail: bool, groups: tuple) -> None:
        """Mooncake thread: prefill wrote groups [lo, hi) of pages [start, end)."""
        with self.loader.condition:
            epoch = self.loader.gate.epoch
            self.prefill_start = (
                start if self.prefill_start is None else min(self.prefill_start, start)
            )
            self.known_end = min(self.known_end, self.prefill_start)
            self.known_log = [
                (min(known, self.prefill_start), when) for known, when in self.known_log
            ]
            # Anything read before this report may hold a partial object.
            keep = max(start, self.first_page)
            inflight = self.inflight
            for group in range(*groups):
                if start < end:
                    self.ready[group] = merge_ranges(self.ready[group], start, end)
                    self.ready_log[group].append((start, end, epoch))
                self.tail_ready[group] = self.tail_ready[group] or tail
                if (
                    inflight is not None
                    and inflight[0] <= group < inflight[1]
                    and keep < inflight[3]
                ):
                    self.versions[group] += 1
                if keep < self.group_loaded[group]:
                    self.versions[group] += 1
                    self.group_loaded[group] = keep
                    self.done = False
            if self.tail_epoch is None and all(self.tail_ready):
                self.tail_epoch = epoch
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

    def next_action(
        self, now: float, slice_pages: int, limit: float = math.inf
    ) -> Optional[tuple]:
        """The next operation, made only of work due before epoch ``limit``."""
        if self.closed or self.failed is not None or self.inflight is not None:
            return None
        if self.object_keys is None:
            return ("prepare",)
        loaded = self.loaded
        if not self.checked:
            if loaded < self.full_pages:
                if self.due_epoch >= limit:
                    return None
                return ("exists", loaded, self.full_pages)
            self.checked = True
        read = self._next_read(slice_pages, math.inf)
        if read is not None:
            if limit != math.inf:
                read = self._next_read(slice_pages, limit)
            return None if read is None else ("get", *read)
        if loaded < self.full_pages:
            # Pages neither prefill nor the first EXISTS covered, such as an
            # older PUT that finished after that query.
            ready = self.ready[self.group_loaded.index(loaded)]
            hole_end = next(
                (left for left, _ in ready if left > loaded),
                self.full_pages if self.prefill_done else None,
            )
            if hole_end is None or now < self.next_hole_check:
                return None
            if self.hole_epoch is None:
                self.hole_epoch = self.loader.gate.epoch
            return ("exists", loaded, hole_end) if self.hole_epoch < limit else None
        if self.has_tail and all(self.tail_ready) and not self.tail_loaded:
            return ("tail",) if self.tail_epoch < limit else None
        return None

    def _readable_until(self, group: int, limit: float) -> int:
        """End of the pages this group may read, counting work due before ``limit``."""
        if limit == math.inf:
            known, ready = self.known_end, self.ready[group]
        else:
            known = max(
                (end for end, epoch in self.known_log if epoch < limit),
                default=self.first_page,
            )
            ready = []
            for start, end, epoch in self.ready_log[group]:
                if epoch < limit:
                    ready = merge_ranges(ready, start, end)
        loaded = self.group_loaded[group]
        return min(max(known, covered_until(ready, loaded)), self.full_pages)

    def _next_read(self, slice_pages: int, limit: float) -> Optional[tuple]:
        """Lowest pages first; the next groups join if they can read them too."""
        count = len(self.groups)
        until = [self._readable_until(group, limit) for group in range(count)]
        readable = [
            group for group in range(count) if until[group] > self.group_loaded[group]
        ]
        if not readable:
            return None
        lo = min(readable, key=lambda group: (self.group_loaded[group], group))
        first = self.group_loaded[lo]
        end = min(until[lo], first + slice_pages)
        hi = lo + 1
        while hi < count and self.group_loaded[hi] == first and until[hi] >= end:
            hi += 1
        return lo, hi, first, end

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
        groups: Sequence[tuple[int, int]],
        clock: Callable[[], float] = time.monotonic,
    ):
        self.backend = backend
        self.gate = gate
        self.slice_pages = slice_pages
        self.groups = list(groups)
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
            job.due_epoch = self.gate.epoch
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
        limit = self.gate.limit()
        due = False
        for job in ordered:
            action = job.next_action(now, self.slice_pages)
            if action is None:
                continue
            if action[0] == "prepare":
                return lambda job=job: self._prepare(job), 0
            due = True
            if limit is None:
                break
            if limit != math.inf:
                # Only work already due when this grant's boundary passed.
                action = job.next_action(now, self.slice_pages, limit)
                if action is None:
                    continue
            self.gate.start()
            if action[0] == "exists":
                _, first, last = action
                return lambda job=job, a=first, b=last: self._exists(job, a, b), 0
            versions = tuple(job.versions)
            if action[0] == "tail":
                pages = job.full_pages
                job.inflight = (0, len(job.groups), pages, pages, versions)
                return lambda job=job: self._get_tail(job), 0
            _, lo, hi, first, last = action
            job.inflight = (lo, hi, first, last, versions)
            return lambda job=job: self._get(job), 0
        if not due:
            self.gate.idle()
            return None, _IDLE_WAIT_S
        self.gate.defer(limit)
        return None, _GATED_WAIT_S

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
        finally:
            self.gate.release()
        with self.condition:
            end = first + hits
            # A later check targets pages prefill did not write.
            if not job.checked and job.prefill_start is not None:
                end = min(end, job.prefill_start)
            job.checked = True
            job.next_hole_check = self.clock() + _HOLE_RECHECK_S
            job.hole_epoch = None
            job.known_end = max(job.known_end, end)
            job.known_log.append((end, self.gate.epoch))

    def _get(self, job: HandoffJob) -> None:
        # Only this thread sets or clears ``inflight``.
        _, _, first, last, _ = job.inflight
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
        lo, hi = job.inflight[:2]
        layers = (job.groups[lo][0], job.groups[hi - 1][1])
        tag = f"{job.rid}@{first}:{layers[0]}"
        error = None
        try:
            self.backend.handoff_get(
                page_keys, slots, tag=tag, first_page=first, layer_range=layers
            )
        except BaseException as caught:
            error = caught
        self.gate.release()
        with self.condition:
            lo, hi, start, end, versions = job.inflight
            job.inflight = None
            if error is not None:
                self._fail(job, error)
            elif start == end:
                job.tail_loaded = True
            else:
                for group in range(lo, hi):
                    if versions[group] == job.versions[group]:
                        job.group_loaded[group] = max(job.group_loaded[group], end)
            finished = error is None and job._finished()
            if finished and not job.done:
                job.done = True
                self.gate.hold(_DECODE_START_HOLD_S)
            self.backend.handoff_applied(
                tag, ok=error is None, loaded_pages=job.loaded, page_size=job.page_size
            )
            self.condition.notify_all()
