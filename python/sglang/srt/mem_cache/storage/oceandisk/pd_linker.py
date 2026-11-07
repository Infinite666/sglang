# SPDX-License-Identifier: Apache-2.0
"""OceanDisk linker for PD disaggregation.

Selected only with ``--disaggregation-linker-notify``. Mixed deployments keep
using ``OceanDiskDirectLinker`` unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.oceandisk.oceandisk_direct_linker import (
    OceanDiskDirectLinker,
)
from sglang.srt.mem_cache.storage.oceandisk.pd_decode_loader import (
    DecodeHandoffLoader,
    DecodeLoadGate,
)
from sglang.srt.mem_cache.unified_cache.pd_handoff import (
    handoff_page_hashes,
    layer_groups,
    tail_page_key,
)
from sglang.srt.observability import ucl_trace

logger = logging.getLogger(__name__)

# Fields of ``OceanDiskDirectLinker._namespace``.
_NS_TP_SIZE = 3
_NS_CP_SIZE = 5


class OceanDiskPDLinker(OceanDiskDirectLinker):
    def __init__(self, server_args, params, *, components, storage=None):
        super().__init__(server_args, params, components=components, storage=storage)
        if not self.pool_group.rank_replicated:
            raise ValueError("Linker notifications require rank-replicated DSA caches.")
        mode = server_args.disaggregation_mode
        self._namespace = self._shared_namespace(server_args, mode)
        self.decode_gate = None
        self.handoff_loader = None
        self._handoff_batch = 0
        self._indexer = self.pools[PoolName.INDEXER]
        # Prefill reports, and decode reads, each layer group of a page range.
        self.handoff_layer_groups = layer_groups(
            self.num_layers, envs.SGLANG_DISAGGREGATION_LINKER_NOTIFY_LAYERS.get()
        )
        # Prefill EXISTS is the mixed one: TP1 queries, the other ranks
        # report every page and the caller's MIN yields TP1's result.
        if mode == "prefill":
            if not self.layerwise_save_enabled:
                raise ValueError(
                    "Linker notifications require layerwise_save=true on prefill."
                )
        elif mode == "decode":
            slice_pages = envs.SGLANG_DISAGGREGATION_LINKER_GET_PAGES.get()
            self._init_indexer_scratch(slice_pages, params)
            self.decode_gate = DecodeLoadGate(
                envs.SGLANG_DISAGGREGATION_LINKER_GET_DECODE_INTERVAL.get()
            )
            self.handoff_loader = DecodeHandoffLoader(
                self, self.decode_gate, slice_pages, self.handoff_layer_groups
            )
            self.handoff_loader.start()
        if ucl_trace.ON:
            self._untraced_exists = self.storage.exists_many
            self.storage.exists_many = self._traced_exists

    def _init_indexer_scratch(self, slice_pages: int, params) -> None:
        """Give every page of a decode GET its own Indexer read target.

        Storage I/O is 512-byte aligned, so an 8448-byte Indexer page is read as
        8704 bytes. Aimed at the real row, the padding lands on the next physical
        page: another request's, or one of this GET that completed first. Reads
        target spare rows past the allocator's range instead, two rows per page
        so the padding stays in the slot, and only the logical row is then copied
        to the request. KV pages are already multiples of 512 bytes.
        """
        for component in self.pools[PoolName.KV].buffer_meta:
            for _, _, size in component:
                if size % 512:
                    raise ValueError(f"OceanDisk KV page of {size} bytes is unaligned.")
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        pool = getattr(kvcache, "full_kv_pool", kvcache)
        scratch_pages = (pool.index_buf_size - pool.size) // self.page_size
        if scratch_pages < 2 * slice_pages + 2:
            raise ValueError(
                f"Decode needs {2 * slice_pages + 2} Indexer scratch rows, "
                f"got {scratch_pages}."
            )
        rows = min(buffer.shape[0] for buffer in self._indexer.get_hybrid_pool_buffer())
        # Even rows sit on 512-byte boundaries: a row is 256 bytes past one.
        self._scratch_first_row = (rows - scratch_pages + 1) & ~1
        self._scratch_pages = slice_pages

    def _scratch_slots(self, count: int) -> tuple[list[int], torch.Tensor]:
        """Rows and token slots of the Indexer read targets of ``count`` pages."""
        if count > self._scratch_pages:
            raise ValueError(f"OceanDisk GET of {count} pages exceeds its scratch.")
        rows = [self._scratch_first_row + 2 * index for index in range(count)]
        page = self.page_size
        slots = (torch.tensor(rows, dtype=torch.int64)[:, None] * page).expand(
            count, page
        ) + torch.arange(page, dtype=torch.int64)
        return rows, slots.reshape(-1)

    def _indexer_publish_plan(self, rows, scratch_rows, layer_range) -> tuple:
        """Built ahead so nothing blocks behind a running GET."""
        target, source = (
            torch.tensor(values, dtype=torch.int64, device=self.device)
            for values in (rows, scratch_rows)
        )
        return target, source, self._indexer_buffers(*layer_range)

    def _indexer_buffers(self, first_layer: int, end_layer: int) -> list:
        """Indexer tensors of layers [first_layer, end_layer), as GET maps them."""
        buffers = []
        for layer in range(first_layer, end_layer):
            mapped = self._indexer.layer_mapping[layer]
            for index in (mapped,) if isinstance(mapped, int) else mapped:
                buffers.extend(
                    component[index] for component in self._indexer.components
                )
        return buffers

    def _publish_indexer(self, plan) -> None:
        """Copy whole logical rows, which end where the next page starts."""
        target, source, buffers = plan
        for buffer in buffers:
            buffer[target] = buffer[source]

    def _shared_namespace(self, server_args, mode):
        """One page object must carry the same key from prefill and decode.

        P is attention TP=1 x CP, D is attention TP x DP: both describe the
        same instance-wide TP layout.
        """
        namespace = list(self._namespace)
        tp_size, cp_size = namespace[_NS_TP_SIZE], namespace[_NS_CP_SIZE]
        if mode == "prefill":
            if tp_size != 1:
                raise ValueError("Linker notifications require prefill attention TP=1.")
            namespace[_NS_TP_SIZE] = cp_size
            namespace[_NS_CP_SIZE] = 1
        elif getattr(server_args, "enable_dp_attention", False):
            namespace[_NS_TP_SIZE] = tp_size * server_args.dp_size
        return tuple(namespace)

    def _traced_exists(self, keys, *, timings=None):
        parts = {} if timings is None else timings
        start = ucl_trace.now()
        try:
            return self._untraced_exists(keys, timings=parts)
        finally:
            ucl_trace.span(
                "od.exists",
                start,
                keys=len(keys),
                put_lock_ms=parts.get("put_lock_ms"),
                put_sync_ms=parts.get("put_sync_ms"),
                route_ms=parts.get("exists_route_ms"),
                upload_ms=parts.get("exists_metadata_ms"),
                native_ms=parts.get("exists_native_ms"),
            )

    # ---- decode loader backend ----

    def loader_context(self):
        return torch.cuda.device(self.device)

    def handoff_keys(self, job):
        """Page hashes, their object keys and the tail page key of one request."""
        hashes = handoff_page_hashes(
            job.req, self.page_size, job.is_bigram, job.full_pages
        )
        tail_key = None
        if job.has_tail:
            tail_key = tail_page_key(
                hashes,
                job.req.origin_input_ids[
                    job.full_pages * self.page_size : job.total_len
                ],
            )
        return hashes, [self._object_key(key) for key in hashes], tail_key

    def handoff_exists(self, object_keys) -> int:
        with torch.cuda.stream(self.storage._get_stream):
            bitmap = self.storage.exists_many(list(object_keys))
        return next((i for i, hit in enumerate(bitmap) if not hit), len(bitmap))

    def handoff_get(
        self, page_keys, slots, *, tag: str, first_page: int, layer_range
    ) -> None:
        """One blocking GET of layers [first, end) into slots the request owns."""
        import od_utils

        self._handoff_batch += 1
        batch = self._handoff_batch
        first_layer, end_layer = layer_range
        if ucl_trace.ON:
            pages = [first_page, first_page + len(page_keys)]
            span = [first_layer, end_layer]
            ucl_trace.emit("linker.load_submit", tag, pages=pages, layers=span)
            ucl_trace.emit(
                "od.get.enqueue",
                batch=batch,
                tags=[tag],
                pages=[len(page_keys)],
                layers=span,
            )
            ucl_trace.emit("od.get.dequeue", batch=batch)
            ucl_trace.emit("od.get.dep_ready", batch=batch)
        indices = torch.from_numpy(np.asarray(slots, dtype=np.int64))
        expanded = self.pool_group.resolve_transfers(
            [
                PoolTransfer(
                    name=PoolName.KV, keys=list(page_keys), device_indices=indices
                )
            ],
            allow_partial=True,
            allow_missing_kv=True,
        )
        indexer = next(item for item in expanded if item.name == PoolName.INDEXER)
        rows = self._indexer.prepare_locations(indexer.host_indices)
        scratch_rows, scratch_slots = self._scratch_slots(len(rows))
        layers = self._get_transfer_layers(
            [
                replace(item, host_indices=scratch_slots)
                if item.name == PoolName.INDEXER
                else item
                for item in expanded
            ]
        )[first_layer:end_layer]
        storage = self.storage
        stream = storage._get_stream
        if ucl_trace.ON:
            ucl_trace.emit(
                "od.get.ranges", batch=batch, ranges=len(layers[0]), copy_ms=0.0
            )
        ok = False
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            try:
                publish_plan = self._indexer_publish_plan(
                    rows, scratch_rows, layer_range
                )
                cpu, gpu = storage._get_metadata_views(len(layers), len(layers[0]))
                storage._pack_layer_batches(layers, output=cpu.numpy())
                gpu.copy_(cpu, non_blocking=True)
                events = storage._get_layer_events[: len(layers)]
                for event in events:
                    event.record(stream)
                if ucl_trace.ON:
                    ucl_trace.emit("od.get.packed")
                gpu_start = ucl_trace.gpu_mark(stream) if ucl_trace.GPU else None
                native_start = ucl_trace.now() if ucl_trace.ON else 0
                storage._run_io(
                    lambda: od_utils.batch_dispatch_get(
                        gpu,
                        gpu.shape[1],
                        cpu,
                        storage._output,
                        [event.cuda_event for event in events],
                        stream.stream_id,
                        self.device.index,
                        0,
                        len(layers),
                        True,
                        False,
                    ),
                    stream,
                )
                if ucl_trace.ON:
                    ucl_trace.span("od.native", native_start, op="get", wait_us=0)
                gpu_end = ucl_trace.gpu_mark(stream) if ucl_trace.GPU else None
                self._publish_indexer(publish_plan)
                # Only this thread uses the GET stream on decode.
                stream.synchronize()
                if gpu_end is not None:
                    ucl_trace.gpu_span("gpu.get", gpu_start, gpu_end)
                ok = True
            except BaseException:
                stream.synchronize()
                raise
            finally:
                if ucl_trace.ON:
                    ucl_trace.emit("od.get.done", batch=batch, ok=ok)
                    ucl_trace.emit("uc.load.consumed", tag, batch=batch, ok=ok)

    def sync_puts(self) -> None:
        """Offload worker thread: wait for every PUT submitted so far."""
        self.storage.record_save_event().synchronize()

    def handoff_applied(self, tag, *, ok, loaded_pages, page_size) -> None:
        if ucl_trace.ON:
            ucl_trace.emit(
                "linker.load_applied",
                tag,
                ok=ok,
                loaded_len=loaded_pages * page_size,
                loaded_layers=0,
            )

    # ---- lifecycle ----

    def reset(self) -> None:
        if self.handoff_loader is not None:
            self.handoff_loader.drain()
        super().reset()
        if ucl_trace.ON:
            # The report pairs GET batches per reset generation.
            ucl_trace.emit("od.reset")

    def close(self) -> None:
        if self._closed:
            return
        if self.handoff_loader is not None:
            self.handoff_loader.stop()
        super().close()
