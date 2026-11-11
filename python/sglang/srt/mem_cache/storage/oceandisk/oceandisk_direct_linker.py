# SPDX-License-Identifier: Apache-2.0
"""Direct GPU prefix-cache storage using the GIDS CUDA operators."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from queue import Empty, Queue
from weakref import WeakValueDictionary

import numpy as np
import torch
from sglang.srt.distributed import get_tp_group
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.oceandisk.oceandisk_storage import (
    OceanDiskRange,
    OceanDiskStorage,
    encode_key,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import get_memory, get_model

logger = logging.getLogger(__name__)


class LayerwiseSaveSession:
    """One forward's puts, completed by the offload worker and reaped by UC."""

    def __init__(self, linker, layers, keys, phase="prefill"):
        self.linker = linker
        self.keys = keys
        self.phase = phase
        self.next_layer = 0
        self.on_submitted = None
        self.release_source = None
        self.completion = Future()
        self.finish_event = None
        self.aborted = False
        self.put_failed = False
        self.ready_events = []
        self.prepare_ms = 0.0
        # The offload worker prepares metadata before submitting the first
        # layer. Keep this session's descriptors alive until then.
        self.layers = layers
        self.prepared = None
        for key in keys:
            linker._layerwise_receipts[key] = self.completion
        linker._put_started(id(self), len(keys))

    def __enter__(self):
        self.linker._layerwise_pending_keys.update(self.keys)
        return self

    def save_layer(self, layer):
        if self.linker.offload_owner:
            event = self.linker._ready_event()
            self.ready_events.append(event)
            self.linker.offload_queue.put((self, layer, event))
        self.next_layer += 1

    def __exit__(self, exc_type, exc, tb):
        self.aborted = exc_type is not None or self.next_layer != self.linker.num_layers
        # The FIFO keeps the completion task after this session's layer tasks.
        # Even aborted forwards retain source pages until queued I/O is drained.
        self.linker.offload_queue.put(self)
        callback, self.on_submitted = self.on_submitted, None
        if callback is not None:
            callback()
        if exc_type is None and self.aborted:
            raise RuntimeError(
                f"OceanDisk saved {self.next_layer}/{self.linker.num_layers} layers."
            )
        return False

    def finish(self, success):
        """Called on the scheduler thread after TP-wide completion consensus."""
        self.linker._layerwise_pending_keys.difference_update(self.keys)
        if not success:
            for key in self.keys:
                if self.linker._layerwise_receipts.get(key) is self.completion:
                    self.linker._layerwise_receipts.pop(key)
        if self.release_source is not None:
            self.release_source()
            self.release_source = None
        self.prepared = None
        self.layers = None
        self.ready_events.clear()


class LayerWiseLoadCounter:
    """Resolve layers only after every physical pool in that layer is readable."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future | None]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int, events=()) -> None:
        self.futures[index][layer].set_result(events)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if future is not None and not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None or futures[threshold] is None:
            return
        try:
            for event in futures[threshold].result():
                event.synchronize()
            futures[threshold] = None  # Subsequent accesses to this layer are ready.
        except BaseException as error:
            raise RuntimeError("OceanDisk layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)

    def reset(self) -> None:
        self.producer_index = self.consumer_index = -1
        self.futures.clear()


class OceanDiskDirectLinker(UnifiedCacheLinker):
    def __init__(self, server_args, params, *, components, storage=None):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=params.token_to_kv_pool_allocator.get_kvcache(),
            page_size=params.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        self.page_size = params.page_size
        if self.num_layers <= 0:
            raise ValueError("OceanDisk requires at least one transfer layer.")
        self._validate_dsa_layout()
        buffers = [
            buffer
            for name in (PoolName.KV, PoolName.INDEXER)
            for buffer in self.pools[name].get_hybrid_pool_buffer()
        ]
        if not buffers:
            raise ValueError("OceanDisk has no device buffers.")
        self.device = buffers[0].device
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        tp_rank, tp_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        replicated = self.pool_group.rank_replicated
        # Interleave CP materializes full MLA KV and Indexer caches on each
        # rank. Match the legacy controller: retain one shared remote copy.
        shared_cp_cache = (
            replicated
            and params.attn_cp_size > 1
            and getattr(server_args, "cp_strategy", None) == "interleave"
        )
        self._exists_tp_group = None
        # Temporary experiment for a shared cache spanning the full model TP group.
        if shared_cp_cache:
            model_tp_group = get_tp_group()
            if params.attn_cp_size == model_tp_group.world_size:
                self._exists_tp_group = model_tp_group
        storage_cp_rank = 0 if shared_cp_cache else params.attn_cp_rank
        self.offload_owner = (not replicated or tp_rank == 0) and (
            not shared_cp_cache or params.attn_cp_rank == 0
        )
        config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        namespace = config.get("namespace", get_model().model_path)
        if not isinstance(namespace, str) or not namespace:
            raise ValueError(
                "OceanDisk namespace must be a nonempty model/version identifier."
            )
        self._namespace = (
            "sglang-oceandisk-pool-pages",
            namespace,
            params.page_size,
            tp_size,
            None if replicated else tp_rank,
            params.attn_cp_size,
            storage_cp_rank,
            params.pp_size,
            params.pp_rank,
        )
        self.storage = (
            OceanDiskStorage(
                config,
                self.device,
                buffers,
            )
            if storage is None
            else storage
        )
        if storage is None:
            self.storage.reserve_get_metadata(self.num_layers)
        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.load_queue = Queue()
        self.completed_loads: Queue[list[str]] = Queue()
        self.offload_queue = Queue()
        self.offload_results: Queue[bool] = Queue()
        self._closed = False
        # Requests and queued UC acknowledgements own completion receipts.
        self._layerwise_receipts = WeakValueDictionary()
        self._layerwise_pending_keys = set()
        self._init_put_progress()
        self.layerwise_save_enabled = config.get("layerwise_save", True)
        # DSA interleave gathers both caches before the per-layer save hook.
        # Reuse the shared namespace and CP0 writer selected above; planning
        # uses the full ScheduleBatch token history and request-to-slot map.
        if self.layerwise_save_enabled and (
            params.pp_size != 1
            or (params.attn_cp_size != 1 and not shared_cp_cache)
            or getattr(server_args, "speculative_algorithm", None)
        ):
            raise ValueError(
                "OceanDisk layerwise_save requires PP=1, no speculative decoding, "
                "and CP=1 or DSA interleave CP with rank-replicated caches."
            )
        if storage is None and self.layerwise_save_enabled and self.offload_owner:
            self.storage.reserve_layerwise_put_metadata(self.num_layers)
        self._load_thread = threading.Thread(
            target=self._load_worker, daemon=True, name=f"oceandisk-load-tp{tp_rank}"
        )
        self._offload_thread = threading.Thread(
            target=self._offload_worker,
            daemon=True,
            name=f"oceandisk-offload-tp{tp_rank}",
        )
        self._load_thread.start()
        self._offload_thread.start()

    def _validate_dsa_layout(self) -> None:
        if set(self.pools) != {PoolName.KV, PoolName.INDEXER}:
            raise ValueError(
                "The GIDS DSA operator requires KV and Indexer pools only."
            )
        counts = []
        self._storage_offsets = {}
        self._get_pool_layouts = {}
        for name, entry in self.pools.items():
            if not entry.packed:
                raise ValueError("OceanDisk requires packed=True for page objects.")
            raw_offset = padded_offset = 0
            offsets = {}
            for component in entry.buffer_meta:
                for _, _, size in component:
                    offsets[raw_offset] = padded_offset
                    raw_offset += size
                    padded_offset += (size + 511) & ~511
            self._storage_offsets[name] = offsets
            layer_counts = []
            get_layout = []
            for layer in range(self.num_layers):
                meta = entry.get_prepared_layer_range_meta([0], layer)
                if meta is None or not meta[1][0]:
                    raise ValueError("Each DSA layer needs KV and Indexer fragments.")
                fragments = [int(size) for size in meta[1][0]]
                layer_counts.append(len(fragments))
                mapped = entry.layer_mapping[layer]
                indices = (mapped,) if isinstance(mapped, int) else tuple(mapped)
                get_layout.append([
                    (base, stride, size, offsets[component_offsets[index]])
                    for component, component_offsets in zip(
                        entry.buffer_meta, entry._component_offsets, strict=True
                    )
                    for index in indices
                    for base, stride, size in (component[index],)
                ])
            self._get_pool_layouts[name] = np.asarray(get_layout, dtype=np.int64)
            counts.extend(layer_counts)
        if len(set(counts)) != 1:
            raise ValueError(
                "GIDS requires paired KV/Indexer IOs and equal IO counts across layers."
            )

    def _object_key(self, page_key) -> int:
        return encode_key((self._namespace, page_key))

    def _page_fragments(
        self, name, page_keys, locations, layer, *, object_keys=None
    ) -> list[OceanDiskRange]:
        """Map a pool/page object's layer to local ranges and remote offsets."""
        if not page_keys:
            return []
        if object_keys is None:
            object_keys = [self._object_key(key) for key in page_keys]
        ptrs, sizes, offsets = self.pools[name].get_prepared_layer_range_meta(
            locations, layer
        )
        ranges = []
        storage_offsets = self._storage_offsets[name]
        for key, pointers, lengths, positions in zip(
            object_keys, ptrs, sizes, offsets, strict=True
        ):
            for ptr, size, storage_offset in zip(
                pointers, lengths, positions, strict=True
            ):
                ranges.append(
                    OceanDiskRange(
                        key, int(ptr), int(size), storage_offsets[storage_offset]
                    )
                )
        return ranges

    def _init_put_progress(self):
        self._put_progress_lock = threading.Lock()
        self._pending_puts = set()

    def _put_started(self, task_id, pages):
        if not self.offload_owner:
            return
        with self._put_progress_lock:
            self._pending_puts.add(task_id)

    def pending_offload_tasks(self):
        """Queued or active PUT sessions not yet confirmed complete."""
        with self._put_progress_lock:
            return len(self._pending_puts)

    def _put_finished(self, task_id, success):
        if not self.offload_owner:
            return
        with self._put_progress_lock:
            self._pending_puts.remove(task_id)

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        self._ensure_open()
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in expanded if transfer.name == PoolName.KV)
        group = getattr(self, "_exists_tp_group", None)
        if group is None:
            end = self._lookup_local_prefix(rid, kv)
        else:
            if group.rank_in_group == 1:
                try:
                    end = self._lookup_local_prefix(rid, kv)
                except Exception:
                    # Raising would leave the other ranks in the caller's MIN.
                    logger.exception(
                        "OceanDisk EXISTS failed on TP1 for %s; treating as a miss", rid
                    )
                    end = 0
            else:
                # The caller's MIN across ranks yields TP1's result.
                end = len(kv.keys)
        return list(range(1, end + 1))

    def _lookup_local_prefix(self, rid, kv) -> int:
        keys = [self._object_key(key) for key in kv.keys]
        bitmap = self.storage.exists_many(keys)
        end = next((i for i, hit in enumerate(bitmap) if not hit), len(bitmap))
        return end

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        self._ensure_open()
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"OceanDisk load already queued for {rid!r}.")
        self.pending_loads[rid] = expanded
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        # Wrapper has already published the tree; cancellation requires rollback.
        return False

    def _ready_event(self):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    def start_layer_wise_loading(self) -> int:
        self._ensure_open()
        if not self.pending_loads:
            return -1
        # Preserve published requests if event creation/recording fails.
        event = self._ready_event()
        index = self.layer_done_counter.update_producer()
        pending, self.pending_loads = self.pending_loads, {}
        try:
            event.synchronize()
            continuation = None
            layers = self._get_transfer_layers(
                [t for transfers in pending.values() for t in transfers]
            )
            if layers and len(layers[0]):
                continuation = self.storage.get_layers(
                    layers,
                    on_layer_submitted=lambda layer, events: self.layer_done_counter.complete(
                        index, layer, events
                    ),
                    defer_remaining=True,
                )
            if continuation is None:
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(index, layer)
        except BaseException as error:
            self.layer_done_counter.fail(index, error)
            self.completed_loads.put(list(pending))
            raise
        self.load_queue.put((index, pending, continuation))
        return index

    def _get_transfer_layers(self, transfers):
        """Build GET key/address/size/offset arrays without per-page objects."""
        cpu_indices = {}
        object_keys = {}
        pools = {PoolName.KV: [], PoolName.INDEXER: []}
        for transfer in transfers:
            source = id(transfer.host_indices)
            if source not in cpu_indices:
                cpu_indices[source] = transfer.host_indices.detach().to(
                    device="cpu", dtype=torch.int64
                )
            entry = self.pools[transfer.name]
            locations = np.asarray(
                entry.prepare_locations(cpu_indices[source]), dtype=np.int64
            )
            if len(locations) != len(transfer.keys):
                raise ValueError("OceanDisk page keys and locations must match.")
            keys = []
            for key in transfer.keys:
                if key not in object_keys:
                    object_keys[key] = self._object_key(key)
                keys.append(object_keys[key])
            # The assembler's mapping, component order and remote offsets are
            # authoritative, including layers with more than one fragment.
            layout = self._get_pool_layouts[transfer.name]
            base, stride, sizes, remote = np.moveaxis(layout, -1, 0)
            fragments = layout.shape[1]
            values = np.empty(
                (self.num_layers, len(keys), fragments, 4), dtype=np.int64
            )
            values[..., 0] = np.asarray(keys, dtype=np.int64)[None, :, None]
            values[..., 1] = base[:, None, :] + locations[None, :, None] * stride[:, None, :]
            values[..., 2] = sizes[:, None, :]
            values[..., 3] = remote[:, None, :]
            pools[transfer.name].append(values.reshape(self.num_layers, -1, 4))
        ordered = pools[PoolName.KV] + pools[PoolName.INDEXER]
        if not ordered:
            return []
        matrix = np.concatenate(ordered, axis=1)
        return list(matrix)

    def _transfer_layers(self, transfers, *, skip_scheduled=False, object_keys=None):
        # A page uses the same object key across layers and physical pools.
        if object_keys is None:
            object_keys = {}
        cpu_indices = {}
        prepared = []
        for transfer in transfers:
            source = id(transfer.host_indices)
            if source not in cpu_indices:
                cpu_indices[source] = transfer.host_indices.detach().to(
                    device="cpu", dtype=torch.int64
                )
            # KV and Indexer share slots, but use token rows and page rows
            # respectively. Convert the slots once, then map each pool on CPU.
            locations = self.pools[transfer.name].prepare_locations(cpu_indices[source])
            page_keys, encoded_keys, page_locations = [], [], []
            for page_key, location in zip(transfer.keys, locations, strict=True):
                if page_key not in object_keys:
                    object_keys[page_key] = self._object_key(page_key)
                key = object_keys[page_key]
                if skip_scheduled and key in self._layerwise_pending_keys:
                    continue
                page_keys.append(page_key)
                encoded_keys.append(key)
                page_locations.append(location)
            prepared.append((transfer.name, page_keys, encoded_keys, page_locations))
        layers = []
        for layer in range(self.num_layers):
            pools = {PoolName.KV: [], PoolName.INDEXER: []}
            for name, page_keys, encoded_keys, locations in prepared:
                pools[name].extend(
                    self._page_fragments(
                        name,
                        page_keys,
                        locations,
                        layer,
                        object_keys=encoded_keys,
                    )
                )
            layers.append(pools[PoolName.KV] + pools[PoolName.INDEXER])
        return layers

    def begin_layerwise_save(self, transfers, *, phase="prefill"):
        kv_transfers = [t for t in transfers if t.name == PoolName.KV]
        object_keys = dict.fromkeys(
            page_key for transfer in kv_transfers for page_key in transfer.keys
        )
        for page_key in object_keys:
            object_keys[page_key] = self._object_key(page_key)
        keys = list(
            dict.fromkeys(
                key
                for key in object_keys.values()
                if key not in self._layerwise_pending_keys
            )
        )
        if not keys:
            return None
        layers = None
        if self.offload_owner:
            # resolve_transfers accepts one transfer per pool, not per request.
            expanded = [
                pool_transfer
                for transfer in kv_transfers
                for pool_transfer in self.pool_group.resolve_transfers(
                    [transfer], allow_partial=True
                )
            ]
            layers = self._transfer_layers(
                expanded,
                skip_scheduled=True,
                object_keys=object_keys,
            )
        session = LayerwiseSaveSession(self, layers, keys, phase)
        return session

    def get_layerwise_receipts(self, page_keys):
        # Pages saved by one session share a Future; check each session once.
        # Keep None so a page with no receipt still makes its UC ack fail.
        return list(
            dict.fromkeys(
                self._layerwise_receipts.get(self._object_key(key)) for key in page_keys
            )
        )

    def _run_transfers(self, transfers, *, write: bool, counter_index=None) -> None:
        layers = (
            self._transfer_layers(transfers) if write
            else self._get_transfer_layers(transfers)
        )
        if not write and (not layers or not len(layers[0])):
            # Nothing to load: release the gates without touching CUDA.
            if counter_index is not None:
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(counter_index, layer)
            return
        if write:
            if layers and layers[0]:
                self.storage.put_layers(layers)
        else:
            self.storage.get_layers(
                layers,
                on_layer_ready=(
                    None
                    if counter_index is None
                    else lambda layer: self.layer_done_counter.complete(
                        counter_index, layer
                    )
                ),
            )

    def _load_worker(self) -> None:
        with torch.cuda.device(self.device):
            while True:
                task = self.load_queue.get()
                try:
                    if task is None:
                        return
                    index, pending, continuation = task
                    try:
                        if continuation is not None:
                            continuation()
                    except BaseException as error:
                        self.layer_done_counter.fail(index, error)
                        logger.exception("OceanDisk load batch failed")
                    finally:
                        self.completed_loads.put(list(pending))
                finally:
                    self.load_queue.task_done()

    def num_completed_loads(self) -> int:
        return self.completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        rids = self.completed_loads.get_nowait()
        return rids

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        self._ensure_open()
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        if not self.offload_owner:
            self.offload_results.put(True)
        else:
            # Layerwise mode only acknowledges existing saves; it never retries PUT.
            event = None if self.layerwise_save_enabled else self._ready_event()
            if self.layerwise_save_enabled:
                kv = next(t for t in expanded if t.name == PoolName.KV)
                payload = self.get_layerwise_receipts(kv.keys)
            else:
                payload = expanded
                kv = next(t for t in expanded if t.name == PoolName.KV)
                self._put_started(id(event), len(kv.keys))
            self.offload_queue.put((payload, event))
        return True

    def _offload_worker(self) -> None:
        with torch.cuda.device(self.device):
            while True:
                try:
                    task = self.offload_queue.get(timeout=1.0)
                except Empty:
                    continue
                try:
                    if task is None:
                        return
                    if isinstance(task, LayerwiseSaveSession):
                        success = False
                        try:
                            if self.offload_owner:
                                task.finish_event = self.storage.record_save_event()
                                task.finish_event.synchronize()
                                if task.prepared is not None:
                                    self.storage.release_layerwise_put_metadata(task.prepared)
                                    task.prepared = None
                            success = not task.aborted and not task.put_failed
                        except BaseException:
                            logger.exception("OceanDisk layerwise offload failed")
                        finally:
                            self._put_finished(id(task), success)
                            task.completion.set_result(success)
                        continue
                    if len(task) == 3:
                        session, layer, ready_event = task
                        if not session.put_failed:
                            try:
                                if session.prepared is None:
                                    session.prepared = self.storage.prepare_layerwise_put(
                                        session.layers
                                    )
                                    session.layers = None
                                self.storage.submit_put_layer(
                                    layer, session.prepared[layer], ready_event
                                )
                            except BaseException:
                                session.put_failed = True
                                logger.exception("OceanDisk layer %d PUT failed", layer)
                        del session
                        continue
                    transfers, event = task
                    success = False
                    try:
                        if self.layerwise_save_enabled:
                            # Session completion is ahead of this task in the same FIFO.
                            success = all(
                                receipt is not None and receipt.result()
                                for receipt in transfers
                            )
                        else:
                            event.synchronize()
                            self._run_transfers(transfers, write=True)
                            success = True
                    except BaseException:
                        logger.exception("OceanDisk offload failed")
                    self.offload_results.put(success)
                    if not self.layerwise_save_enabled:
                        self._put_finished(id(event), success)
                    del transfers
                finally:
                    self.offload_queue.task_done()
                    del task

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OceanDisk linker is closed.")

    def reset(self) -> None:
        # Even queued loads target published nodes; drain them before tree reset.
        if self.pending_loads:
            self.start_layer_wise_loading()
        self.load_queue.join()
        self.offload_queue.join()
        for results in (self.completed_loads, self.offload_results):
            while True:
                try:
                    results.get_nowait()
                except Empty:
                    break
        self.layer_done_counter.reset()
        self._layerwise_receipts.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        self._closed = True
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self._load_thread.join()
        self._offload_thread.join()
        self.storage.close()
