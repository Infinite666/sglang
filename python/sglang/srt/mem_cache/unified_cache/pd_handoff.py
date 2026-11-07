# SPDX-License-Identifier: Apache-2.0
"""PD-only hooks on ``UnifiedCacheLinkerWrapper``; never installed in mixed mode.

Every page of a prompt reaches decode through OceanDisk. Prefill adds the
partial last page to its final save session and reports each finished session;
Mooncake only carries these reports, the first token and the aux data.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.observability import ucl_trace

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )

logger = logging.getLogger(__name__)

TAIL_KEY_PREFIX = "pd-tail:"


def handoff_page_hashes(
    req: Req, page_size: int, is_bigram: bool, num_pages: int
) -> list[str]:
    """Hash chain of the first ``num_pages`` full pages, identical on P and D."""
    if num_pages == 0:
        return []
    key = RadixKey(
        req.origin_input_ids,
        req.extra_key,
        cache_salt=req.cache_salt,
        is_bigram=is_bigram,
    )[: num_pages * page_size]
    return get_storage_hash_str(key, prior_hash=None, page_size=page_size)


def tail_page_key(page_hashes: Sequence[str], tail_tokens) -> str:
    """Storage identity of a partial last page.

    It is never inserted into a radix tree and never probed with EXISTS, and
    its prefix keeps it apart from a real page holding the same tokens.
    """
    tokens = np.asarray(list(tail_tokens), dtype=np.int64).tobytes()
    last_hash = page_hashes[-1] if page_hashes else ""
    return f"{TAIL_KEY_PREFIX}{last_hash}:{hashlib.sha256(tokens).hexdigest()}"


class _ReceiptJoin:
    """Report one page range once every session writing it has finished."""

    def __init__(self, sender, start: int, end: int, tail: bool, receipts: list):
        self.sender = sender
        self.start = start
        self.end = end
        self.tail = tail
        self.left = len(receipts)
        self.success = True
        self.lock = threading.Lock()
        for receipt in receipts:
            receipt.add_done_callback(self._done)

    def _done(self, receipt) -> None:
        try:
            success = bool(receipt.result())
        except BaseException:
            logger.exception("OceanDisk save receipt failed")
            success = False
        with self.lock:
            self.left -= 1
            self.success = self.success and success
            finished = self.left == 0
            result = self.success
        if not finished:
            return
        if result:
            self.sender.notify_linker_ready(
                self.start, self.end, self.tail, receipt=id(receipt)
            )
        else:
            self.sender.fail_linker("OceanDisk PUT did not complete successfully")


def _trace_session_done(session_id: int, pages: int, receipt) -> None:
    ucl_trace.emit(
        "od.put.session_done",
        session=session_id,
        pages=pages,
        ok=receipt.result(),
        receipt=id(receipt),
    )


class PrefillHandoff:
    bypass_match = False

    def __init__(self, wrapper: UnifiedCacheLinkerWrapper):
        self.wrapper = wrapper

    def add_tail_pages(self, batch: ScheduleBatch, transfers, completed) -> None:
        """Plan the partial last page of every request ending in this forward."""
        cache = self.wrapper.cache
        page = cache.page_size
        for req, end in zip(batch.reqs, batch.seq_lens_cpu.tolist(), strict=True):
            sender = getattr(req, "disagg_kv_sender", None)
            if (
                not hasattr(sender, "linker_begin")
                or end != len(req.origin_input_ids)
                or end % page == 0
            ):
                continue
            full_pages = end // page
            hashes = getattr(req, "_uc_layerwise_hashes", None) or []
            if len(hashes) < full_pages:
                # The last forward may hold no new full page.
                hashes = handoff_page_hashes(
                    req, page, cache.tree_core.is_eagle, full_pages
                )
            tokens = cache.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, full_pages * page : end
            ]
            # Slots past ``end`` are unused, but the object is one whole page.
            first_slot = tokens[:1] - tokens[:1] % page
            transfers.append(
                PoolTransfer(
                    name=PoolName.KV,
                    keys=[
                        tail_page_key(
                            hashes[:full_pages],
                            req.origin_input_ids[full_pages * page : end],
                        )
                    ],
                    device_indices=first_slot
                    + torch.arange(page, device=tokens.device, dtype=tokens.dtype),
                )
            )
            completed.append((req, full_pages * page))

    def bind_session(self, transfers, completed, session) -> None:
        """Report the pages of this forward as their PUT sessions finish."""
        backend = self.wrapper.cache_linker
        page = self.wrapper.cache.page_size
        if session is not None and ucl_trace.ON:
            ucl_trace.emit(
                "uc.save.begin",
                session=id(session),
                phase=session.phase,
                rids=[req.rid for req, _ in completed],
                pages=[len(transfer.keys) for transfer in transfers],
                prepare_ms=round(session.prepare_ms, 3),
            )
        if not backend.offload_owner:
            return
        if session is not None and ucl_trace.ON:
            session.completion.add_done_callback(
                lambda receipt, session_id=id(session), pages=len(session.keys): (
                    _trace_session_done(session_id, pages, receipt)
                )
            )
        for transfer, (req, length) in zip(transfers, completed, strict=True):
            sender = getattr(req, "disagg_kv_sender", None)
            if not hasattr(sender, "linker_begin"):
                continue
            tail = transfer.keys[0].startswith(TAIL_KEY_PREFIX)
            end = length // page
            start = end if tail else end - len(transfer.keys)
            sender.linker_begin()
            receipts = backend.get_layerwise_receipts(transfer.keys)
            if any(receipt is None for receipt in receipts):
                sender.fail_linker("OceanDisk save has no completion receipt")
                continue
            _ReceiptJoin(sender, start, end, tail, receipts)


class DecodeHandoff:
    """Decode reads what prefill reports, not what the tree finds in storage."""

    bypass_match = True
