"""CPU tests for the decode-side OceanDisk loader and its decode gate."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from sglang.srt.mem_cache.storage.oceandisk.pd_decode_loader import (
    DecodeHandoffLoader,
    DecodeLoadGate,
    covered_until,
    group_span,
    merge_ranges,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PAGE = 4
ALL = (0, 1)  # The only layer group of LoaderCase.


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeBackend:
    def __init__(self, stored_pages=0):
        self.stored_pages = stored_pages
        self.reads = []
        self.layer_ranges = []
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

    def handoff_get(self, page_keys, slots, *, tag, first_page, layer_range):
        if self.on_read is not None:
            self.on_read()
        if self.fail_reads:
            raise RuntimeError("GET failed")
        self.reads.append((first_page, list(page_keys), np.asarray(slots).tolist()))
        self.layer_ranges.append(tuple(layer_range))

    def handoff_applied(self, tag, *, ok, loaded_pages, page_size):
        pass


def decode_batch():
    return SimpleNamespace(
        global_num_tokens=[1, 0, 0, 0], forward_mode=SimpleNamespace()
    )


class LoaderCase(unittest.TestCase):
    groups = [(0, 4)]

    def setUp(self):
        self.clock = FakeClock()
        self.gate = DecodeLoadGate(interval=4, clock=self.clock)
        self.backend = FakeBackend()
        self.loader = DecodeHandoffLoader(
            self.backend, self.gate, slice_pages=2, groups=self.groups, clock=self.clock
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

    def steps(self, count):
        for _ in range(count):
            self.gate.update(decode_batch())

    def read_pages(self):
        return [(first, len(keys)) for first, keys, _ in self.backend.reads]

    def read_layers(self):
        return [
            (first, len(keys), layers)
            for (first, keys, _), layers in zip(
                self.backend.reads, self.backend.layer_ranges
            )
        ]


class TestRanges(unittest.TestCase):
    def test_merge_keeps_gaps(self):
        ranges = merge_ranges([], 8, 12)
        ranges = merge_ranges(ranges, 0, 4)
        self.assertEqual(ranges, [(0, 4), (8, 12)])
        self.assertEqual(covered_until(ranges, 0), 4)
        self.assertEqual(covered_until(ranges, 4), 4)
        self.assertEqual(merge_ranges(ranges, 4, 8), [(0, 12)])

    def test_group_span_needs_whole_groups(self):
        groups = [(0, 3), (3, 4)]
        self.assertEqual(group_span(groups, 0, 4), (0, 2))
        self.assertEqual(group_span(groups, 0, 3), (0, 1))
        self.assertEqual(group_span(groups, 3, 4), (1, 2))
        self.assertIsNone(group_span(groups, 0, 2))
        self.assertIsNone(group_span(groups, 3, 3))


class TestDecodeLoadGate(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.gate = DecodeLoadGate(interval=4, clock=self.clock)

    def steps(self, count):
        for _ in range(count):
            self.gate.update(decode_batch())

    def take(self, due=0):
        """Start one operation due since epoch ``due``, as the loader does."""
        limit = self.gate.limit()
        if limit is None or due >= limit:
            self.gate.defer(limit)
            return False
        self.gate.start()
        return True

    def test_idle_decode_allows_every_slice(self):
        self.assertTrue(self.take())
        self.gate.release()
        self.assertTrue(self.take())

    def test_decode_releases_one_slice_per_interval(self):
        released = []
        for step in range(1, 9):
            self.gate.update(decode_batch())
            released.append((step, self.take(), self.take()))
            self.gate.release()
        self.assertEqual(
            [step for step, first, _ in released if first],
            [4, 8],
        )
        self.assertFalse(any(second for _, _, second in released))

    def test_waiting_loader_keeps_its_grant(self):
        self.gate.update(decode_batch())
        self.assertFalse(self.take())
        self.steps(4)
        self.assertTrue(self.take())
        self.assertFalse(self.take())

    def test_idle_loader_banks_no_grant(self):
        self.steps(5)
        due = self.gate.epoch
        self.assertFalse(self.take(due))
        self.steps(2)
        self.assertFalse(self.take(due))
        self.steps(1)
        self.assertTrue(self.take(due))

    def test_work_due_before_a_boundary_takes_it_after_the_operation(self):
        self.gate.update(decode_batch())
        self.assertFalse(self.take())
        self.steps(3)
        self.assertTrue(self.take())
        self.steps(4)
        self.gate.release()
        self.assertTrue(self.take())

    def test_work_due_after_a_boundary_waits_for_the_next(self):
        self.assertTrue(self.take())
        self.steps(4)  # Passes while the operation runs.
        due = self.gate.epoch
        self.gate.release()
        self.assertFalse(self.take(due))
        self.steps(4)
        self.assertTrue(self.take(due))

    def test_grants_do_not_pile_up(self):
        self.assertTrue(self.take())
        self.steps(12)
        self.gate.release()
        self.assertTrue(self.take())
        self.gate.release()
        self.assertFalse(self.take())

    def test_idle_loader_drops_an_unused_grant(self):
        self.assertTrue(self.take())
        self.steps(4)
        self.gate.release()
        self.gate.idle()
        self.assertFalse(self.take())

    def test_hold_keeps_the_grant(self):
        self.gate.update(decode_batch())
        self.assertFalse(self.take())
        self.steps(3)
        self.gate.hold(0.5)
        self.assertFalse(self.take())
        self.clock.now += 1
        self.assertTrue(self.take())

    def test_grant_does_not_outlive_decode(self):
        self.gate.update(decode_batch())
        self.assertFalse(self.take())
        self.steps(3)
        self.gate.update(None)
        self.clock.now += 1
        self.gate.update(decode_batch())
        self.assertFalse(self.take())

    def test_last_forward_keeps_the_gate_closed_briefly(self):
        self.gate.update(decode_batch())
        self.gate.update(None)
        self.assertFalse(self.take())
        self.clock.now += 1
        self.assertTrue(self.take())

    def test_idle_batch_is_not_decode(self):
        batch = SimpleNamespace(global_num_tokens=[0, 0, 0, 0])
        self.gate.update(batch)
        self.assertTrue(self.take())

    def test_without_dp_sync_the_forward_mode_decides(self):
        batch = SimpleNamespace(
            global_num_tokens=None,
            forward_mode=SimpleNamespace(is_decode=lambda: True),
        )
        self.gate.update(batch)
        self.assertFalse(self.take())
        self.gate.update(batch)
        self.gate.update(batch)
        self.gate.update(batch)
        self.assertTrue(self.take())
        self.assertFalse(self.take())


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
        job.on_ready(3, 5, False, ALL)
        self.drive()
        self.assertEqual(job.loaded, 5)
        self.assertFalse(job.done)
        job.on_ready(5, 5, True, ALL)
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
        job.on_ready(3, 5, False, ALL)
        self.assertFalse(job.done)
        self.assertEqual(job.loaded, 3)
        self.clock.now += 1
        self.drive()
        self.assertEqual(self.read_pages()[-1], (3, 2))
        self.assertTrue(job.done)

    def test_ready_during_a_get_discards_that_get(self):
        self.backend.stored_pages = 4
        job = self.job(total_len=4 * PAGE)
        self.backend.on_read = lambda: (
            job.on_ready(1, 4, False, ALL) if len(self.backend.reads) == 0 else None
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
        job.on_ready(1, 4, False, ALL)
        self.assertEqual(job.loaded, 2)
        self.clock.now += 1
        self.drive()
        self.assertEqual(self.read_pages(), [(2, 2), (2, 2)])
        self.assertTrue(job.done)

    def test_exists_cannot_pass_a_known_prefill_page(self):
        self.backend.stored_pages = 4
        job = self.job(total_len=4 * PAGE)
        job.on_ready(2, 4, False, ALL)
        self.drive()
        self.assertEqual(self.backend.exists_calls, [(0, 4)])
        self.assertEqual(self.read_pages(), [(0, 2), (2, 2)])
        self.assertTrue(job.done)

    def test_gap_before_prefill_pages_is_checked_again(self):
        job = self.job(total_len=4 * PAGE)
        self.drive()
        job.on_ready(2, 4, False, ALL)
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
        self.gate.update(decode_batch())
        self.drive()
        self.assertEqual(self.backend.exists_calls, [])
        self.steps(3)
        self.drive()
        self.assertEqual(self.backend.exists_calls, [(0, 2)])
        self.assertEqual(self.backend.reads, [])
        self.steps(4)
        self.drive()
        self.assertEqual(self.read_pages(), [(0, 2)])
        self.assertTrue(job.done)

    def test_report_mid_interval_waits_for_the_next_boundary(self):
        self.backend.stored_pages = 2
        job = self.job(total_len=4 * PAGE)
        self.drive()
        self.assertEqual(job.loaded, 2)
        # Step 4 finds nothing to read, so no grant waits for the report.
        self.steps(5)
        self.drive()
        job.on_ready(2, 4, False, ALL)
        self.drive()
        self.assertEqual(self.read_pages(), [(0, 2)])
        self.steps(3)
        self.drive()
        self.assertEqual(self.read_pages(), [(0, 2), (2, 2)])

    def test_grant_due_during_a_get_starts_the_next_one(self):
        self.backend.stored_pages = 4
        self.job(total_len=4 * PAGE)
        self.gate.update(decode_batch())
        self.drive()
        self.steps(3)
        self.drive()
        self.assertEqual(self.backend.exists_calls, [(0, 4)])
        self.steps(4)
        self.backend.on_read = lambda: (
            self.steps(4) if not self.backend.reads else None
        )
        self.drive()
        self.assertEqual(self.read_pages(), [(0, 2), (2, 2)])

    def reach_first_get(self):
        """EXISTS at step 4; step 8 releases the grant for the first GET."""
        self.gate.update(decode_batch())
        self.drive()
        self.steps(3)
        self.drive()
        self.steps(4)

    def test_report_during_a_get_waits_for_the_boundary_after_it(self):
        self.backend.stored_pages = 2
        job = self.job(total_len=4 * PAGE)
        self.reach_first_get()

        def during_get():
            self.steps(4)  # Step 12 passes while the GET runs.
            job.on_ready(2, 4, False, ALL)

        self.backend.on_read = during_get
        self.drive()
        self.assertEqual(self.read_pages(), [(0, 2)])
        self.backend.on_read = None
        self.steps(4)
        self.drive()
        self.assertEqual(self.read_pages(), [(0, 2), (2, 2)])

    def test_grant_after_a_get_goes_to_pages_due_before_it(self):
        self.backend.stored_pages = 4
        job = self.job(total_len=6 * PAGE)
        self.reach_first_get()

        def during_get():
            if not self.backend.reads:
                self.steps(4)
                job.on_ready(4, 6, False, ALL)

        self.backend.on_read = during_get
        self.drive()
        # [2, 4) was known before step 12; prefill's [4, 6) came after it.
        self.assertEqual(self.read_pages(), [(0, 2), (2, 2)])
        self.steps(4)
        self.drive()
        self.assertEqual(self.read_pages(), [(0, 2), (2, 2), (4, 2)])

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
        second.on_ready(0, 4, False, ALL)
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


class TestLayerGroups(LoaderCase):
    groups = [(0, 3), (3, 4)]

    def test_committed_prefix_reads_every_layer_at_once(self):
        self.backend.stored_pages = 4
        job = self.job(total_len=4 * PAGE)
        self.drive()
        self.assertEqual(self.read_layers(), [(0, 2, (0, 4)), (2, 2, (0, 4))])
        self.assertTrue(job.done)

    def test_first_group_is_read_while_the_last_is_written(self):
        job = self.job(total_len=4 * PAGE)
        self.drive()
        job.on_ready(0, 4, False, (0, 1))
        self.drive()
        self.assertEqual(self.read_layers(), [(0, 2, (0, 3)), (2, 2, (0, 3))])
        self.assertEqual(job.group_loaded, [4, 0])
        self.assertFalse(job.done)
        # Pages the next group still waits for are not a hole.
        self.assertEqual(self.backend.exists_calls, [(0, 4)])
        job.on_ready(0, 4, False, (1, 2))
        self.drive()
        self.assertEqual(self.read_layers()[2:], [(0, 2, (3, 4)), (2, 2, (3, 4))])
        self.assertTrue(job.done)

    def test_groups_reported_together_share_one_get(self):
        job = self.job(total_len=4 * PAGE)
        self.drive()
        job.on_ready(0, 4, False, (0, 1))
        job.on_ready(0, 4, False, (1, 2))
        self.drive()
        self.assertEqual(self.read_layers(), [(0, 2, (0, 4)), (2, 2, (0, 4))])
        self.assertTrue(job.done)

    def test_report_rereads_only_its_group(self):
        self.backend.stored_pages = 4
        job = self.job(total_len=4 * PAGE)
        self.drive()
        job.on_ready(2, 4, False, (0, 1))
        self.assertEqual(job.group_loaded, [2, 4])
        self.clock.now += 1
        self.drive()
        self.assertEqual(self.read_layers()[-1], (2, 2, (0, 3)))

    def test_report_during_a_shared_get_keeps_the_other_group(self):
        self.backend.stored_pages = 2
        job = self.job(total_len=2 * PAGE)
        self.backend.on_read = lambda: (
            job.on_ready(0, 2, False, (0, 1)) if not self.backend.reads else None
        )
        self.drive()
        self.assertEqual(self.read_layers(), [(0, 2, (0, 4)), (0, 2, (0, 3))])
        self.assertEqual(job.group_loaded, [2, 2])

    def test_tail_waits_for_every_group_and_reads_all_layers(self):
        job = self.job(total_len=2 * PAGE + 1)
        self.drive()
        job.on_ready(0, 2, False, (0, 1))
        job.on_ready(0, 2, False, (1, 2))
        self.drive()
        self.assertFalse(job.done)
        job.on_ready(2, 2, True, (0, 2))
        self.drive()
        self.assertTrue(job.done)
        self.assertEqual(self.read_layers()[-1], (2, 1, (0, 4)))


if __name__ == "__main__":
    unittest.main()
