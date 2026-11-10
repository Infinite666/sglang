"""CPU tests for the decode-side OceanDisk loader and its decode gate."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from sglang.srt.mem_cache.storage.oceandisk import pd_decode_loader
from sglang.srt.mem_cache.storage.oceandisk.pd_decode_loader import (
    DecodeHandoffLoader,
    DecodeLoadGate,
    covered_until,
    merge_ranges,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PAGE = 4


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeBackend:
    def __init__(self, stored_pages=0):
        self.stored_pages = stored_pages
        self.reads = []
        self.exists_calls = []
        self.fail_reads = False
        self.on_read = None

    def loader_context(self):
        return nullcontext()

    def handoff_keys(self, job):
        hashes = [f"page-{index}" for index in range(job.full_pages)]
        return hashes, list(range(job.full_pages)), "tail" if job.has_tail else None

    def handoff_exists(self, object_keys):
        self.exists_calls.append((object_keys[0], object_keys[-1] + 1))
        hits = 0
        for key in object_keys:
            if key >= self.stored_pages:
                break
            hits += 1
        return hits

    def handoff_get(self, page_keys, slots, *, tag, first_page):
        if self.on_read is not None:
            self.on_read()
        if self.fail_reads:
            raise RuntimeError("GET failed")
        self.reads.append((first_page, list(page_keys), np.asarray(slots).tolist()))

    def handoff_applied(self, tag, *, ok, loaded_pages, page_size):
        pass


def decode_batch():
    return SimpleNamespace(
        global_num_tokens=[1, 0, 0, 0], forward_mode=SimpleNamespace()
    )


class LoaderCase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.gate = DecodeLoadGate(interval=4, clock=self.clock)
        self.backend = FakeBackend()
        self.loader = DecodeHandoffLoader(
            self.backend, self.gate, slice_pages=2, clock=self.clock
        )

    def job(self, total_len, loaded_pages=0):
        job = self.loader.new_job(
            rid="request",
            room=7,
            req=SimpleNamespace(),
            slots=np.arange(total_len, dtype=np.int32),
            total_len=total_len,
            page_size=PAGE,
            loaded_pages=loaded_pages,
        )
        self.loader.submit(job)
        return job

    def drive(self):
        """Run the loader's work on this thread until it has nothing to do."""
        for _ in range(100):
            with self.loader.condition:
                work, _ = self.loader._next_work()
            if work is None:
                return
            work()
        self.fail("The loader did not settle.")

    def read_pages(self):
        return [(first, len(keys)) for first, keys, _ in self.backend.reads]


class TestRanges(unittest.TestCase):
    def test_merge_keeps_gaps(self):
        ranges = merge_ranges([], 8, 12)
        ranges = merge_ranges(ranges, 0, 4)
        self.assertEqual(ranges, [(0, 4), (8, 12)])
        self.assertEqual(covered_until(ranges, 0), 4)
        self.assertEqual(covered_until(ranges, 4), 4)
        self.assertEqual(merge_ranges(ranges, 4, 8), [(0, 12)])


class TestDecodeLoadGate(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.gate = DecodeLoadGate(interval=4, clock=self.clock)

    def test_idle_decode_allows_every_slice(self):
        self.assertTrue(self.gate.allow())
        self.assertTrue(self.gate.allow())

    def test_decode_releases_one_slice_per_interval(self):
        released = []
        for step in range(1, 9):
            self.gate.update(decode_batch())
            released.append((step, self.gate.allow(), self.gate.allow()))
        self.assertEqual(
            [step for step, first, _ in released if first],
            [4, 8],
        )
        self.assertFalse(any(second for _, _, second in released))

    def test_grant_survives_until_the_loader_takes_it(self):
        for _ in range(5):
            self.gate.update(decode_batch())
        self.assertTrue(self.gate.allow())
        self.assertFalse(self.gate.allow())

    def test_grants_do_not_pile_up(self):
        for _ in range(12):
            self.gate.update(decode_batch())
        self.assertTrue(self.gate.allow())
        self.assertFalse(self.gate.allow())

    def test_hold_keeps_the_grant(self):
        for _ in range(4):
            self.gate.update(decode_batch())
        self.gate.hold(0.5)
        self.assertFalse(self.gate.allow())
        self.clock.now += 1
        self.assertTrue(self.gate.allow())

    def test_grant_does_not_outlive_decode(self):
        for _ in range(4):
            self.gate.update(decode_batch())
        self.gate.update(None)
        self.clock.now += 1
        self.gate.update(decode_batch())
        self.assertFalse(self.gate.allow())

    def test_last_forward_keeps_the_gate_closed_briefly(self):
        self.gate.update(decode_batch())
        self.gate.update(None)
        self.assertFalse(self.gate.allow())
        self.clock.now += 1
        self.assertTrue(self.gate.allow())

    def test_idle_batch_is_not_decode(self):
        batch = SimpleNamespace(global_num_tokens=[0, 0, 0, 0])
        self.gate.update(batch)
        self.assertTrue(self.gate.allow())

    def test_without_dp_sync_the_forward_mode_decides(self):
        batch = SimpleNamespace(
            global_num_tokens=None,
            forward_mode=SimpleNamespace(is_decode=lambda: True),
        )
        self.gate.update(batch)
        self.gate.update(batch)
        self.gate.update(batch)
        self.gate.update(batch)
        self.assertTrue(self.gate.allow())
        self.assertFalse(self.gate.allow())


class TestHandoffJob(LoaderCase):
    def test_committed_prefix_loads_before_prefill_reports(self):
        self.backend.stored_pages = 5
        job = self.job(total_len=7 * PAGE + 1)
        self.drive()
        self.assertEqual(self.backend.exists_calls, [(0, 7)])
        self.assertEqual(self.read_pages(), [(0, 2), (2, 2), (4, 1)])
        self.assertEqual(job.loaded, 5)
        self.assertFalse(job.done)

    def test_ready_then_tail_finishes_the_job(self):
        self.backend.stored_pages = 3
        job = self.job(total_len=5 * PAGE + 1)
        self.drive()
        job.on_ready(3, 5, False)
        self.drive()
        self.assertEqual(job.loaded, 5)
        self.assertFalse(job.done)
        job.on_ready(5, 5, True)
        self.drive()
        self.assertTrue(job.done)
        first, keys, slots = self.backend.reads[-1]
        self.assertEqual((first, keys), (5, ["tail"]))
        self.assertEqual(slots, [20, 21, 22, 23])

    def test_page_aligned_prompt_has_no_tail(self):
        self.backend.stored_pages = 3
        job = self.job(total_len=3 * PAGE)
        self.drive()
        self.assertTrue(job.done)

    def test_decode_hit_skips_storage(self):
        job = self.job(total_len=3 * PAGE, loaded_pages=3)
        self.drive()
        self.assertTrue(job.done)
        self.assertEqual(self.backend.exists_calls, [])
        self.assertEqual(self.backend.reads, [])

    def test_ready_rereads_a_page_exists_saw_mid_put(self):
        # EXISTS is true once layer 0 lands, while prefill still writes page 3.
        self.backend.stored_pages = 5
        job = self.job(total_len=5 * PAGE)
        self.drive()
        self.assertEqual(job.loaded, 5)
        self.assertTrue(job.done)
        job.on_ready(3, 5, False)
        self.assertFalse(job.done)
        self.assertEqual(job.loaded, 3)
        self.clock.now += 1
        self.drive()
        self.assertEqual(self.read_pages()[-1], (3, 2))
        self.assertTrue(job.done)

    def test_pages_nobody_provides_are_reported_once(self):
        job = self.job(total_len=4 * PAGE)
        job.on_ready(2, 4, False)
        self.drive()
        with self.assertLogs(pd_decode_loader.logger, "WARNING") as logs:
            for wait in (1, 6, 6):
                self.clock.now += wait
                self.drive()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("pages [0, 2)", logs.output[0])
        self.assertEqual(job.loaded, 0)

    def test_ready_during_a_get_discards_that_get(self):
        self.backend.stored_pages = 4
        job = self.job(total_len=4 * PAGE)
        self.backend.on_read = lambda: (
            job.on_ready(1, 4, False) if len(self.backend.reads) == 0 else None
        )
        self.drive()
        # Page 1 was in the discarded slice; page 0 stands alone before prefill's.
        self.assertEqual(self.read_pages(), [(0, 2), (0, 1), (1, 2), (3, 1)])
        self.assertEqual(job.loaded, 4)
        self.assertTrue(job.done)

    def test_ready_below_the_decode_hit_keeps_it(self):
        self.backend.stored_pages = 4
        job = self.job(total_len=4 * PAGE, loaded_pages=2)
        self.drive()
        self.assertEqual(self.read_pages(), [(2, 2)])
        job.on_ready(1, 4, False)
        self.assertEqual(job.loaded, 2)
        self.clock.now += 1
        self.drive()
        self.assertEqual(self.read_pages(), [(2, 2), (2, 2)])
        self.assertTrue(job.done)

    def test_exists_cannot_pass_a_known_prefill_page(self):
        self.backend.stored_pages = 4
        job = self.job(total_len=4 * PAGE)
        job.on_ready(2, 4, False)
        self.drive()
        self.assertEqual(self.backend.exists_calls, [(0, 4)])
        self.assertEqual(self.read_pages(), [(0, 2), (2, 2)])
        self.assertTrue(job.done)

    def test_gap_before_prefill_pages_is_checked_again(self):
        job = self.job(total_len=4 * PAGE)
        self.drive()
        job.on_ready(2, 4, False)
        self.drive()
        self.assertEqual(self.backend.reads, [])
        self.backend.stored_pages = 2
        self.clock.now += 1
        self.drive()
        self.assertEqual(self.backend.exists_calls[-1], (0, 2))
        self.assertEqual(self.read_pages(), [(0, 2), (2, 2)])
        self.assertTrue(job.done)

    def test_prefill_success_rechecks_pages_it_never_reported(self):
        job = self.job(total_len=2 * PAGE)
        self.drive()
        self.assertFalse(job.done)
        self.backend.stored_pages = 2
        self.clock.now += 1
        self.drive()
        self.assertEqual(self.backend.reads, [])
        job.prefill_finished()
        self.drive()
        self.assertTrue(job.done)

    def test_get_failure_is_recorded(self):
        self.backend.stored_pages = 2
        self.backend.fail_reads = True
        job = self.job(total_len=2 * PAGE)
        with self.assertLogs(level="ERROR"):
            self.drive()
        self.assertEqual(job.failed, "GET failed")
        self.assertFalse(job.busy)
        self.assertFalse(job.done)

    def test_decode_pauses_reads_until_it_ends(self):
        self.backend.stored_pages = 2
        job = self.job(total_len=2 * PAGE)
        self.gate.update(decode_batch())
        self.drive()
        self.assertEqual(self.backend.exists_calls, [])
        self.gate.update(None)
        self.drive()
        self.assertEqual(self.backend.exists_calls, [])
        self.clock.now += 1
        self.drive()
        self.assertEqual(job.loaded, 2)

    def test_exists_and_get_each_take_one_grant(self):
        self.backend.stored_pages = 2
        job = self.job(total_len=2 * PAGE)
        for _ in range(4):
            self.gate.update(decode_batch())
        self.drive()
        self.assertEqual(self.backend.exists_calls, [(0, 2)])
        self.assertEqual(self.backend.reads, [])
        for _ in range(4):
            self.gate.update(decode_batch())
        self.drive()
        self.assertEqual(self.read_pages(), [(0, 2)])
        self.assertTrue(job.done)

    def test_finished_request_holds_reads_for_its_first_step(self):
        self.backend.stored_pages = 2
        self.job(total_len=2 * PAGE)
        self.drive()
        other = self.job(total_len=2 * PAGE)
        self.drive()
        self.assertEqual(other.loaded, 0)
        self.clock.now += 1
        self.drive()
        self.assertEqual(other.loaded, 2)

    def test_reported_requests_go_first(self):
        self.backend.stored_pages = 4
        first = self.job(total_len=4 * PAGE)
        second = self.job(total_len=4 * PAGE)
        self.backend.on_read = lambda: self.backend.reads.append(
            (first.rid, ["order"], [id(self.loader.jobs)])
        )
        second.on_ready(0, 4, False)
        with self.loader.condition:
            work, _ = self.loader._next_work()
        work()
        self.assertIsNotNone(second.object_keys)
        self.assertIsNone(first.object_keys)

    def test_closed_job_stops_reading(self):
        self.backend.stored_pages = 2
        job = self.job(total_len=2 * PAGE)
        job.close()
        self.drive()
        self.assertEqual(self.backend.reads, [])
        self.assertEqual(self.loader.jobs, [])

    def test_drain_drops_every_job(self):
        job = self.job(total_len=2 * PAGE)
        self.loader.drain()
        self.assertTrue(job.closed)
        self.assertEqual(self.loader.jobs, [])


if __name__ == "__main__":
    unittest.main()
