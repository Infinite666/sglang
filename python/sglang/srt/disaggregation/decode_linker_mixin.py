"""Decode-side gate over the OceanDisk handoff loader."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.environ import envs
from sglang.srt.observability import ucl_trace

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest

logger = logging.getLogger(__name__)


def new_linker_job(tree_cache, req_to_token_pool, decode_req, input_len: int):
    """Describe a request's pages; the loader computes keys off the scheduler."""
    req = decode_req.req
    page = tree_cache.page_size
    loader = tree_cache.linker.cache_linker.handoff_loader
    return loader.new_job(
        rid=req.rid,
        room=req.bootstrap_room,
        req=req,
        slots=req_to_token_pool.req_to_token[req.kv.req_pool_idx, :input_len]
        .cpu()
        .numpy(),
        total_len=input_len,
        page_size=page,
        loaded_pages=len(req.prefix_indices) // page,
        is_bigram=tree_cache.tree_core.is_eagle,
    )


def _record_linker_failure(dr: DecodeRequest, reason: str) -> None:
    receiver = dr.kv_receiver
    with receiver.kv_mgr.failure_lock:
        receiver.kv_mgr.failure_records.setdefault(receiver.bootstrap_room, reason)


class LinkerLoadGatedKVReceiver:
    """Hold Success until this rank has read every page of the request."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        dr = self.decode_req
        poll = dr.kv_receiver.poll()
        job = dr.linker_job
        if ucl_trace.ON and getattr(dr, "_ucl_kv_poll", None) != poll:
            dr._ucl_kv_poll = poll
            ucl_trace.emit(
                "linker.kv_poll", dr.req.rid, state=getattr(poll, "name", poll)
            )
        if job is None:
            return poll
        if poll == KVPoll.Failed or dr.kv_receiver.abort_notified:
            return poll
        if job.failed is None and not job.done:
            elapsed = time.perf_counter() - (
                dr.req.time_stats.decode_transfer_queue_entry_time
            )
            if elapsed >= envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get():
                job.failed = (
                    f"OceanDisk load timed out after {elapsed:.1f}s: "
                    f"pages={job.loaded}/{job.full_pages}"
                )
        if job.failed is not None:
            _record_linker_failure(dr, job.failed)
            return KVPoll.Failed
        if poll == KVPoll.Success and not job.done:
            job.prefill_finished()
            return KVPoll.Transferring
        return poll


def apply_linker_polls(queue: list[DecodeRequest], polls: list, group) -> list:
    """Stop the loader on terminal states; release Failed only once DMA drained.

    A rank can learn a failure from a peer with its own GET still running, and
    its slots are freed as soon as it sees Failed. Polls are identical across
    ranks here, so every rank enters the reduction below for the same requests
    and keeps those requests queued as Transferring until all of them drained.
    """
    failed = []
    for index, (dr, poll) in enumerate(zip(queue, polls, strict=True)):
        job = dr.linker_job
        if job is None or poll not in (KVPoll.Failed, KVPoll.Success):
            continue
        job.close()
        if poll == KVPoll.Failed:
            failed.append(index)
    if not failed:
        return polls
    drained = torch.tensor(
        [not queue[index].linker_job.busy for index in failed],
        dtype=torch.uint8,
        device="cpu",
    )
    torch.distributed.all_reduce(
        drained, op=torch.distributed.ReduceOp.MIN, group=group
    )
    for index, done in zip(failed, drained.tolist(), strict=True):
        if not done:
            polls[index] = KVPoll.Transferring
    return polls
