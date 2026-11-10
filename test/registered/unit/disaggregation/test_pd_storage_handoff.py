"""CPU tests for the PD storage handoff: READY transport, tail page and routing."""

import threading
import unittest
from collections import defaultdict
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode_linker_mixin import (
    LinkerLoadGatedKVReceiver,
    apply_linker_polls,
)
from sglang.srt.disaggregation.mooncake import conn as mooncake
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferInfo,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers import data_parallel_controller
from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.oceandisk.pd_linker import OceanDiskPDLinker
from sglang.srt.mem_cache.unified_cache import pd_handoff
from sglang.srt.mem_cache.unified_cache.pd_handoff import (
    TAIL_KEY_PREFIX,
    DecodeHandoff,
    PrefillHandoff,
    tail_page_key,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeJob:
    def __init__(self, full_pages=8):
        self.full_pages = full_pages
        self.reports = []

    def on_ready(self, start, end, tail):
        self.reports.append((start, end, tail))


class TestReadyTransport(unittest.TestCase):
    def setUp(self):
        mgr = self.mgr = object.__new__(MooncakeKVManager)
        mgr.linker_notify = True
        mgr.linker_num_layers = 3
        mgr.linker_senders, mgr.linker_receivers = {}, {}
        mgr._staging_outstanding = defaultdict(int)
        mgr.request_status = {7: KVPoll.Transferring}
        mgr.failure_records, mgr.failure_lock = {}, threading.Lock()
        mgr.kv_status_msg_carries_reason = True
        mgr.kv_status_msg_tag = None
        mgr._prefill_unique_rank = lambda: 0
        self.peer = SimpleNamespace(
            is_dummy=False,
            required_dst_info_num=1,
            endpoint="127.0.0.1",
            dst_port=1234,
            mooncake_session_id="decode",
        )
        mgr.transfer_infos = {7: {"decode": self.peer}}
        sender = self.sender = object.__new__(MooncakeKVSender)
        sender.kv_mgr, sender.bootstrap_room = mgr, 7
        sender.linker_lock = threading.RLock()
        sender.linker_ready, sender.linker_pending = [], 0
        sender.linker_aux_done = sender.linker_concluded = False
        mgr.linker_senders[7] = sender
        receiver = self.receiver = object.__new__(MooncakeKVReceiver)
        receiver.kv_mgr, receiver.bootstrap_room = mgr, 7
        receiver.session_id = "decode"
        receiver.linker_job = self.job = FakeJob()
        mgr.linker_receivers[7] = receiver
        self.messages = []

        def deliver(address, parts, **kwargs):
            self.messages.append(parts)
            if parts[0] == b"LINKER_READY" and 7 in mgr.linker_receivers:
                receiver.receive_linker_ready(parts)

        mgr._send_multipart_locked = Mock(side_effect=deliver)

    def finish_aux(self):
        self.mgr.conclude_transfer(bootstrap_room=7, status=KVPoll.Success)

    def test_aux_alone_concludes_when_no_put_runs(self):
        self.finish_aux()
        self.assertEqual(self.mgr.request_status[7], KVPoll.Success)

    def test_success_waits_for_a_running_put(self):
        self.sender.linker_begin()
        self.finish_aux()
        self.assertEqual(self.mgr.request_status[7], KVPoll.Transferring)
        self.sender.notify_linker_ready(0, 8, False)
        self.assertEqual(self.job.reports, [(0, 8, False)])
        self.assertEqual(self.mgr.request_status[7], KVPoll.Success)

    def test_put_finished_before_aux_still_waits_for_aux(self):
        self.sender.linker_begin()
        self.sender.notify_linker_ready(0, 8, False)
        self.assertEqual(self.mgr.request_status[7], KVPoll.Transferring)
        self.finish_aux()
        self.assertEqual(self.mgr.request_status[7], KVPoll.Success)

    def test_tail_flag_reaches_decode(self):
        self.sender.linker_begin()
        self.sender.notify_linker_ready(8, 8, True)
        self.assertEqual(self.job.reports, [(8, 8, True)])

    def test_ready_before_bootstrap_is_kept(self):
        infos = self.mgr.transfer_infos.pop(7)
        self.sender.linker_begin()
        self.sender.notify_linker_ready(0, 8, False)
        self.assertEqual(self.job.reports, [])
        self.mgr.transfer_infos[7] = infos
        self.sender.flush_linker_ready()
        self.assertEqual(self.job.reports, [(0, 8, False)])

    def test_dummy_peer_gets_no_ready(self):
        self.peer.is_dummy = True
        self.sender.linker_begin()
        self.sender.notify_linker_ready(0, 8, False)
        self.assertEqual(self.messages, [])

    def test_send_exception_becomes_native_failure(self):
        self.sender.fail_linker = Mock()
        self.mgr._send_multipart_locked.side_effect = OSError("gone")
        self.sender.linker_begin()
        self.sender.notify_linker_ready(0, 8, False)
        self.sender.fail_linker.assert_called_once()

    def test_old_sender_cannot_publish_into_reused_room(self):
        self.mgr.linker_senders[7] = object()
        self.sender.linker_begin()
        self.sender.notify_linker_ready(0, 8, False)
        self.assertEqual(self.messages, [])

    def test_out_of_range_ready_fails_the_room(self):
        self.mgr.record_failure = Mock()
        self.mgr.update_status = Mock()
        self.receiver.receive_linker_ready(
            [b"LINKER_READY", b"7", b"decode", b"0", b"9", b"0"]
        )
        self.mgr.update_status.assert_called_once_with(7, KVPoll.Failed)
        self.assertEqual(self.job.reports, [])

    def test_wrong_session_is_rejected(self):
        with self.assertRaises(ValueError):
            self.receiver.receive_linker_ready(
                [b"LINKER_READY", b"7", b"other", b"0", b"8", b"0"]
            )

    def test_metadata_marker(self):
        base = [
            b"7",
            b"127.0.0.1",
            b"1234",
            b"decode:1",
            b"",
            b"3",
            b"",
            b"1",
            b"160",
            b"",
        ]
        self.assertFalse(TransferInfo.from_zmq(base).linker_notify)
        info = TransferInfo.from_zmq([*base, b"LINKER_NOTIFY"])
        self.assertTrue(info.linker_notify)
        self.assertFalse(info.is_dummy)
        self.assertEqual(len(info.dst_kv_indices), 0)
        self.assertEqual(info.dst_state_indices, [])

    def test_aux_goes_over_tcp(self):
        self.mgr.enable_custom_mem_pool = False
        self.mgr.custom_mem_pool_type = None
        self.mgr.send_aux_tcp = Mock(return_value=0)
        self.mgr.send_aux(SimpleNamespace(), 1, [])
        self.mgr.send_aux_tcp.assert_called_once()

    def initialize(self, *, notify, state_types):
        self.mgr.disaggregation_mode = DisaggregationMode.DECODE
        args = SimpleNamespace(
            kv_layer_ids=[], kv_data_ptrs=list(range(78)), state_types=state_types
        )
        with (
            patch.object(CommonKVManager, "__init__", return_value=None),
            patch.object(MooncakeKVManager, "init_engine"),
            patch.object(MooncakeKVManager, "register_buffer_to_engine"),
            patch.object(MooncakeKVManager, "start_decode_thread"),
            patch.object(
                mooncake,
                "get_observability",
                return_value=SimpleNamespace(enable_trace=False),
            ),
            envs.SGLANG_DISAGG_STAGING_BUFFER.override(False),
        ):
            self.mgr.__init__(
                args,
                DisaggregationMode.DECODE,
                SimpleNamespace(disaggregation_linker_notify=notify),
                is_mla_backend=True,
            )

    def test_manager_counts_layers_only_with_notifications(self):
        self.initialize(notify=True, state_types=[StateType.DSA])
        self.assertEqual(self.mgr.linker_num_layers, 78)
        self.initialize(notify=False, state_types=[StateType.DSA_TAIL])
        self.assertEqual(self.mgr.linker_num_layers, 0)

    def test_state_outside_oceandisk_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unsupported transfer state"):
            self.initialize(notify=True, state_types=[StateType.DSA_TAIL])


class TestTailPage(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.object(
                pd_handoff,
                "get_storage_hash_str",
                side_effect=lambda key, prior_hash, page_size: [
                    f"p{i}" for i in range(len(key) // page_size)
                ],
            )
        )
        self.sender = SimpleNamespace(
            linker_begin=Mock(), notify_linker_ready=Mock(), fail_linker=Mock()
        )
        self.req = SimpleNamespace(
            rid="request",
            origin_input_ids=list(range(10)),
            extra_key=None,
            cache_salt=None,
            disagg_kv_sender=self.sender,
            kv=SimpleNamespace(req_pool_idx=0),
        )
        self.backend = SimpleNamespace(offload_owner=True)
        self.wrapper = SimpleNamespace(
            cache=SimpleNamespace(
                page_size=4,
                tree_core=SimpleNamespace(is_eagle=False),
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.arange(100, 164).reshape(1, 64)
                ),
            ),
            cache_linker=self.backend,
        )
        self.handoff = PrefillHandoff(self.wrapper)

    def batch(self, end):
        return SimpleNamespace(reqs=[self.req], seq_lens_cpu=torch.tensor([end]))

    def test_tail_is_planned_as_one_whole_page(self):
        transfers, completed = [], []
        self.handoff.add_tail_pages(self.batch(10), transfers, completed)
        self.assertEqual(len(transfers), 1)
        transfer = transfers[0]
        self.assertEqual(transfer.name, PoolName.KV)
        self.assertEqual(
            transfer.keys, [tail_page_key(["p0", "p1"], self.req.origin_input_ids[8:])]
        )
        self.assertTrue(transfer.keys[0].startswith(TAIL_KEY_PREFIX))
        self.assertEqual(transfer.device_indices.tolist(), [108, 109, 110, 111])
        self.assertEqual(completed, [(self.req, 8)])

    def test_prefill_and_decode_derive_the_same_key(self):
        hashes = pd_handoff.handoff_page_hashes(self.req, 4, False, 2)
        self.assertEqual(
            tail_page_key(hashes, np.array(self.req.origin_input_ids[8:10])),
            tail_page_key(["p0", "p1"], self.req.origin_input_ids[8:10]),
        )

    def test_other_tokens_get_another_key(self):
        self.assertNotEqual(
            tail_page_key(["p0"], [1, 2]), tail_page_key(["p0"], [1, 3])
        )

    def test_mid_prompt_forward_has_no_tail(self):
        transfers, completed = [], []
        self.handoff.add_tail_pages(self.batch(8), transfers, completed)
        self.assertEqual(transfers, [])

    def test_page_aligned_prompt_has_no_tail(self):
        self.req.origin_input_ids = list(range(8))
        transfers, completed = [], []
        self.handoff.add_tail_pages(self.batch(8), transfers, completed)
        self.assertEqual(transfers, [])

    def test_fake_warmup_sender_is_skipped(self):
        self.req.disagg_kv_sender = SimpleNamespace()
        transfers, completed = [], []
        self.handoff.add_tail_pages(self.batch(10), transfers, completed)
        self.assertEqual(transfers, [])

    def bind(self, receipts, keys=("p0", "p1"), length=8):
        self.backend.get_layerwise_receipts = Mock(return_value=receipts)
        transfer = PoolTransfer(PoolName.KV, keys=list(keys))
        self.handoff.bind_session([transfer], [(self.req, length)], None)

    def test_session_completion_reports_its_pages(self):
        receipt = Future()
        self.bind([receipt])
        self.sender.linker_begin.assert_called_once()
        self.sender.notify_linker_ready.assert_not_called()
        receipt.set_result(True)
        self.sender.notify_linker_ready.assert_called_once_with(
            0, 2, False, receipt=id(receipt)
        )

    def test_tail_report_names_the_page_boundary(self):
        receipt = Future()
        self.bind([receipt], keys=(f"{TAIL_KEY_PREFIX}x",))
        receipt.set_result(True)
        self.sender.notify_linker_ready.assert_called_once_with(
            2, 2, True, receipt=id(receipt)
        )

    def test_range_waits_for_every_session_writing_it(self):
        first, second = Future(), Future()
        self.bind([first, second])
        first.set_result(True)
        self.sender.notify_linker_ready.assert_not_called()
        second.set_result(True)
        self.sender.notify_linker_ready.assert_called_once()

    def test_failed_session_fails_the_room(self):
        receipt = Future()
        self.bind([receipt])
        receipt.set_result(False)
        self.sender.fail_linker.assert_called_once()
        self.sender.notify_linker_ready.assert_not_called()

    def test_missing_receipt_fails_the_room(self):
        self.bind([None])
        self.sender.fail_linker.assert_called_once()

    def test_only_the_put_owner_reports(self):
        self.backend.offload_owner = False
        self.handoff.bind_session(
            [PoolTransfer(PoolName.KV, keys=["p0"])], [(self.req, 4)], None
        )
        self.sender.linker_begin.assert_not_called()


class TestWrapperHooks(unittest.TestCase):
    def test_mixed_wrapper_has_no_handoff(self):
        self.assertIsNone(UnifiedCacheLinkerWrapper.handoff)

    def test_decode_does_not_probe_storage(self):
        wrapper = object.__new__(UnifiedCacheLinkerWrapper)
        wrapper.handoff = DecodeHandoff()
        result = object()
        self.assertIs(wrapper.match(None, None, result), result)


class TestPdLinker(unittest.TestCase):
    def namespace(self, tp_size, cp_size):
        return ("tag", "model", 64, tp_size, None, cp_size, 0, 1, 0)

    def test_prefill_and_decode_share_page_objects(self):
        prefill = object.__new__(OceanDiskPDLinker)
        prefill._namespace = self.namespace(1, 8)
        decode = object.__new__(OceanDiskPDLinker)
        decode._namespace = self.namespace(2, 1)
        self.assertEqual(
            prefill._shared_namespace(SimpleNamespace(), "prefill"),
            decode._shared_namespace(
                SimpleNamespace(enable_dp_attention=True, dp_size=4), "decode"
            ),
        )

    def test_prefill_needs_attention_tp_one(self):
        prefill = object.__new__(OceanDiskPDLinker)
        prefill._namespace = self.namespace(2, 4)
        with self.assertRaises(ValueError):
            prefill._shared_namespace(SimpleNamespace(), "prefill")

    def scratch_linker(self, rows=20, kv_page_bytes=73728):
        linker = object.__new__(OceanDiskPDLinker)
        linker.device = "cpu"
        linker.page_size = 4
        self.buffer = torch.zeros(rows, 16, dtype=torch.uint8)
        linker._indexer = SimpleNamespace(get_hybrid_pool_buffer=lambda: [self.buffer])
        linker.pools = {
            PoolName.KV: SimpleNamespace(buffer_meta=[[(0, 0, kv_page_bytes)]])
        }
        return linker

    def scratch_params(self, scratch_pages):
        pool = SimpleNamespace(size=400, index_buf_size=400 + scratch_pages * 4)
        return SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool)
        )

    def test_scratch_rows_keep_padding_inside_their_own_slot(self):
        linker = self.scratch_linker(rows=21)
        linker._init_indexer_scratch(3, self.scratch_params(8))
        rows, slots = linker._scratch_slots(3)
        self.assertEqual(rows, [14, 16, 18])
        self.assertEqual(slots[:4].tolist(), [56, 57, 58, 59])
        self.assertEqual(slots[4:8].tolist(), [64, 65, 66, 67])
        self.assertLessEqual(rows[-1] + 2, 21)

    def test_scratch_row_is_never_an_allocatable_row(self):
        linker = self.scratch_linker(rows=20)
        linker._init_indexer_scratch(3, self.scratch_params(8))
        rows, _ = linker._scratch_slots(3)
        self.assertGreaterEqual(rows[0], 20 - 8)

    def test_get_larger_than_scratch_is_refused(self):
        linker = self.scratch_linker()
        linker._init_indexer_scratch(3, self.scratch_params(8))
        with self.assertRaises(ValueError):
            linker._scratch_slots(4)

    def test_missing_scratch_rows_fail_at_startup(self):
        linker = self.scratch_linker()
        with self.assertRaisesRegex(ValueError, "scratch"):
            linker._init_indexer_scratch(3, self.scratch_params(0))

    def test_unaligned_kv_page_fails_at_startup(self):
        linker = self.scratch_linker(kv_page_bytes=73730)
        with self.assertRaisesRegex(ValueError, "unaligned"):
            linker._init_indexer_scratch(3, self.scratch_params(8))

    def test_publish_copies_logical_rows_and_nothing_else(self):
        linker = self.scratch_linker()
        self.buffer[:] = 7
        # Padded reads left bytes in the scratch slots' next rows.
        self.buffer[14] = 1
        self.buffer[15] = 99
        self.buffer[16] = 2
        self.buffer[17] = 99
        linker._publish_indexer(linker._indexer_publish_plan([3, 4], [14, 16]))
        self.assertEqual(self.buffer[3].unique().tolist(), [1])
        self.assertEqual(self.buffer[4].unique().tolist(), [2])
        self.assertEqual(self.buffer[5].unique().tolist(), [7])
        self.assertEqual(self.buffer[2].unique().tolist(), [7])


class TestLoadGatedPoll(unittest.TestCase):
    def setUp(self):
        self.job = SimpleNamespace(
            done=False,
            failed=None,
            loaded=1,
            full_pages=2,
            prefill_finished=Mock(),
            close=Mock(),
            busy=False,
        )
        self.receiver = SimpleNamespace(
            poll=Mock(return_value=KVPoll.Transferring),
            abort_notified=False,
            bootstrap_room=7,
            kv_mgr=SimpleNamespace(failure_lock=threading.Lock(), failure_records={}),
        )
        self.req = SimpleNamespace(
            rid="request",
            time_stats=SimpleNamespace(decode_transfer_queue_entry_time=0.0),
        )
        self.dr = SimpleNamespace(
            kv_receiver=self.receiver, linker_job=self.job, req=self.req
        )
        self.enterContext(
            patch(
                "sglang.srt.disaggregation.decode_linker_mixin.time.perf_counter",
                return_value=1.0,
            )
        )
        self.enterContext(envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.override(300))

    def poll(self):
        return LinkerLoadGatedKVReceiver(self.dr).poll()

    def test_native_poll_passes_through_without_a_job(self):
        self.dr.linker_job = None
        self.receiver.poll.return_value = KVPoll.Success
        self.assertEqual(self.poll(), KVPoll.Success)

    def test_success_waits_for_the_pages(self):
        self.receiver.poll.return_value = KVPoll.Success
        self.assertEqual(self.poll(), KVPoll.Transferring)
        self.job.prefill_finished.assert_called_once()
        self.job.done = True
        self.assertEqual(self.poll(), KVPoll.Success)

    def test_finished_pages_still_wait_for_prefill(self):
        self.job.done = True
        self.assertEqual(self.poll(), KVPoll.Transferring)

    def test_load_failure_is_recorded(self):
        self.job.failed = "GET failed"
        self.assertEqual(self.poll(), KVPoll.Failed)
        self.assertEqual(self.receiver.kv_mgr.failure_records, {7: "GET failed"})

    def test_timeout_fails_an_unfinished_job(self):
        self.req.time_stats.decode_transfer_queue_entry_time = -400.0
        self.assertEqual(self.poll(), KVPoll.Failed)
        self.assertIn("timed out", self.job.failed)

    def apply(self, polls, *, peers_drained=True):
        def all_reduce(tensor, op, group):
            self.assertIs(group, self.group)
            if not peers_drained:
                tensor[:] = 0

        self.group = object()
        with patch.object(
            torch.distributed, "all_reduce", side_effect=all_reduce
        ) as reduce:
            result = apply_linker_polls([self.dr], polls, self.group)
        return result, reduce

    def test_success_stops_the_loader_without_a_collective(self):
        polls, reduce = self.apply([KVPoll.Success])
        self.assertEqual(polls, [KVPoll.Success])
        self.job.close.assert_called_once()
        reduce.assert_not_called()

    def test_running_transfer_keeps_the_loader(self):
        polls, reduce = self.apply([KVPoll.Transferring])
        self.assertEqual(polls, [KVPoll.Transferring])
        self.job.close.assert_not_called()
        reduce.assert_not_called()

    def test_failure_passes_once_every_rank_drained(self):
        polls, reduce = self.apply([KVPoll.Failed])
        self.assertEqual(polls, [KVPoll.Failed])
        self.job.close.assert_called_once()
        reduce.assert_called_once()

    def test_own_running_get_holds_the_failure(self):
        self.job.busy = True
        polls, _ = self.apply([KVPoll.Failed])
        self.assertEqual(polls, [KVPoll.Transferring])
        self.job.close.assert_called_once()

    def test_peer_running_get_holds_the_failure(self):
        polls, _ = self.apply([KVPoll.Failed], peers_drained=False)
        self.assertEqual(polls, [KVPoll.Transferring])

    def test_requests_without_a_job_skip_the_reduction(self):
        self.dr.linker_job = None
        polls, reduce = self.apply([KVPoll.Failed])
        self.assertEqual(polls, [KVPoll.Failed])
        reduce.assert_not_called()


class TestDecodeAffinity(unittest.TestCase):
    def setUp(self):
        controller = self.controller = object.__new__(DataParallelController)
        controller.workers = ["w0", "w1", "w2", "w3"]
        controller._active_workers = [0, 1, 2, 3]
        controller.status = [True] * 4
        controller.round_robin_counter = 0
        controller._linker_affinity = data_parallel_controller.OrderedDict()
        controller._linker_new_session = controller.round_robin_scheduler
        self.sent = self.enterContext(
            patch.object(data_parallel_controller, "sock_send")
        )

    def request(self, key=None, input_ids=(1, 2, 3)):
        return SimpleNamespace(
            routed_dp_rank=None, routing_key=key, input_ids=list(input_ids)
        )

    def destinations(self):
        return [call.args[0] for call in self.sent.call_args_list]

    def test_session_stays_on_its_group(self):
        for key in ("a", "b", "c", "d", "a", "b", "c", "d"):
            self.controller.linker_affinity_scheduler(self.request(key))
        self.assertEqual(
            self.destinations(), ["w0", "w1", "w2", "w3", "w0", "w1", "w2", "w3"]
        )

    def test_prompt_prefix_identifies_a_session_without_a_key(self):
        first = list(range(5000))
        second = [*first, 1, 2, 3]
        other = [9, *first]
        for ids in (first, other, second):
            self.controller.linker_affinity_scheduler(self.request(input_ids=ids))
        self.assertEqual(self.destinations(), ["w0", "w1", "w0"])

    def test_unavailable_group_moves_the_session(self):
        self.controller.linker_affinity_scheduler(self.request("a"))
        self.controller.status[0] = False
        self.controller.linker_affinity_scheduler(self.request("a"))
        self.controller.linker_affinity_scheduler(self.request("a"))
        self.assertEqual(self.destinations(), ["w0", "w1", "w1"])

    def test_skipped_worker_is_not_recorded(self):
        self.controller.status[0] = False
        self.controller.linker_affinity_scheduler(self.request("a"))
        self.controller.linker_affinity_scheduler(self.request("a"))
        self.assertEqual(self.destinations(), ["w1", "w1"])

    def test_configured_method_places_only_the_first_request(self):
        placed = iter([2, 0])
        self.controller._linker_new_session = lambda req: next(placed)
        for key in ("a", "b", "a", "b"):
            self.controller.linker_affinity_scheduler(self.request(key))
        # Later requests of a session are sent here, not by the method.
        self.assertEqual(self.destinations(), ["w2", "w0"])
        self.assertEqual(dict(self.controller._linker_affinity), {"a": 2, "b": 0})

    def test_explicit_dp_rank_wins(self):
        request = self.request("a")
        request.routed_dp_rank = 3
        self.controller.linker_affinity_scheduler(request)
        self.assertEqual(self.destinations(), ["w3"])
        self.assertEqual(len(self.controller._linker_affinity), 0)


if __name__ == "__main__":
    unittest.main()
