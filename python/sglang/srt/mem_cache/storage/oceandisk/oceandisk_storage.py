# SPDX-License-Identifier: Apache-2.0
"""OceanDisk direct-GPU storage with a reserved HiCache interface."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.storage.oceandisk.connection_router import ConnectionRouter

if TYPE_CHECKING:
    import torch

MAX_GIDS_CONNECTIONS = 8  # od/gids_kernel.cu: MAX_CONN_NUM
MAX_GIDS_RANGES = 409600  # od/gids_kernel.cu: MAX_BLOCK_NUM
GET_METADATA_WIDTH = 40960  # od/cache_controller.py: max_block_pram_length
PUT_METADATA_REQUESTS = 100  # Legacy controller: _max_batch_block / layer_num.
PUT_METADATA_ROW_WIDTH = 640 * 6  # Legacy controller: max_block_for_40k.


def gids_config_path(config_dir, device_index: int, use_single_port=None) -> Path:
    """Match gids_kernel.cu: unset USE_SINGLE_PORT defaults to single-port."""
    single = use_single_port is None or use_single_port == "1"
    filename = "gids_config.txt" if single else f"gids_config_{device_index}.txt"
    return Path(config_dir) / filename


def read_connection_count(path: Path) -> int:
    """Fail before native init, which otherwise logs bad config and may succeed."""
    count = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or "ip_config" not in line:
            continue
        name, sep, value = line.partition("=")
        fields = [field.strip() for field in value.split(",")]
        if (
            not raw.startswith("ip_config")
            or len(raw.encode()) >= 256
            or not sep
            or name.strip() != "ip_config"
            or len(fields) != 6
        ):
            raise ValueError(f"Invalid GIDS ip_config at {path}:{number}.")
        try:
            port, gid, roce_port, qp_count = map(int, fields[2:])
        except ValueError as error:
            raise ValueError(
                f"Invalid GIDS connection numbers at {path}:{number}."
            ) from error
        if (
            not fields[0]
            or not fields[1]
            or len(value.split(",")[0].lstrip().encode()) >= 32
            or len(value.split(",")[1].lstrip().encode()) >= 32
            or not 0 < port <= 65535
            or not 0 <= gid < 256
            or roce_port < 0
            or qp_count < 0
        ):
            raise ValueError(f"Invalid GIDS connection fields at {path}:{number}.")
        count += 1
    if not 1 <= count <= MAX_GIDS_CONNECTIONS:
        raise ValueError(
            f"OceanDisk requires 1..{MAX_GIDS_CONNECTIONS} ip_config entries: {path}."
        )
    return count


@dataclass(frozen=True)
class OceanDiskRange:
    key: int
    ptr: int
    size: int
    storage_offset: int = 0


@lru_cache(maxsize=65536)
def encode_key(identity: tuple) -> int:
    """Encode an object identity as a positive int64."""
    data = json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode()
    return (
        int.from_bytes(hashlib.sha256(data).digest()[:8], "big") & ((1 << 63) - 1)
    ) or 1


def resolve_mr(regions: list[tuple[int, int]], ptr: int, size: int) -> tuple[int, int]:
    if size <= 0:
        raise ValueError("OceanDisk ranges must have positive length.")
    for mr_id, (base, length) in enumerate(regions):
        if base <= ptr and ptr + size <= base + length:
            return mr_id, ptr - base
    raise ValueError(f"OceanDisk range ptr={ptr:#x}, size={size} is not registered.")


def pack_io(
    task_id: int,
    ranges: list[OceanDiskRange],
    regions: list[tuple[int, int]],
    connection_indices: tuple[int, ...],
    *,
    resolve=None,
    select_connection=None,
) -> list[int]:
    """Column-major ABI used by od/cache_controller.py, with remote offsets."""
    half = len(ranges) // 2
    if (
        not ranges
        or len(ranges) % 2
        or [r.key for r in ranges[:half]] != [r.key for r in ranges[half:]]
    ):
        raise ValueError("OceanDisk requires matching KV and Indexer halves.")
    mr_offsets = [
        resolve(item.ptr, item.size)
        if resolve is not None
        else resolve_mr(regions, item.ptr, item.size)
        for item in ranges
    ]
    if any(item.storage_offset < 0 for item in ranges):
        raise ValueError("OceanDisk storage offsets must be nonnegative.")
    paired_connections = [
        select_connection(item.key)
        if select_connection is not None
        else connection_indices[item.key % len(connection_indices)]
        for item in ranges[:half]
    ]
    return [
        task_id,
        *[mr for mr, _ in mr_offsets],
        *[item.key for item in ranges],
        *[offset for _, offset in mr_offsets],
        *[item.storage_offset for item in ranges],
        *[item.size for item in ranges],
        *paired_connections,
        *paired_connections,
    ]


class OceanDiskStorage(HiCacheStorage):
    """Direct-buffer I/O with separate load/store streams; aligned ranges can overlap pages."""

    _registration_lock = threading.Lock()
    _initialized = False
    _registered_tensors: list = []

    def __init__(
        self,
        config: dict,
        device: torch.device,
        buffers: list[torch.Tensor],
    ):
        connections = config.get("connection_indices", [])
        if "connection_indices" in config and (
            not isinstance(connections, list)
            or not connections
            or any(
                type(index) is not int or not 0 <= index < MAX_GIDS_CONNECTIONS
                for index in connections
            )
            or len(set(connections)) != len(connections)
        ):
            raise ValueError(
                "OceanDisk connection_indices must be distinct integers in [0, 7]."
            )
        self.connection_indices = tuple(connections)
        self._closed = False
        self._task_ids = itertools.count(1)
        self._get_metadata_cpu = None
        self._get_metadata_gpu = None
        self._get_layer_events = None

        import torch

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The OceanDisk direct linker requires CUDA buffers.")
        buffers = list(buffers)
        self.regions = []
        for buffer in buffers:
            if buffer.device != self.device:
                raise ValueError(
                    "OceanDisk buffers must all be on the linker's CUDA device."
                )
            storage = buffer.untyped_storage()
            region = (int(storage.data_ptr()), int(storage.nbytes()))
            if region[1] and region not in self.regions:
                self.regions.append(region)
        if not self.regions:
            raise ValueError("OceanDisk has no GPU allocations to register.")
        self._build_mr_index()

        library = config.get("op_extension") or os.environ.get("OPEXTENSION")
        if not library:
            raise ValueError(
                "Set OceanDisk op_extension or OPEXTENSION to libop_extension.so."
            )
        # LIBGIDS must be set before load_library: the library constructor dlopens it.
        for key, env in (("core_library", "LIBGIDS"), ("config_dir", "GIDS_DIR")):
            if config.get(key):
                os.environ[env] = os.fspath(config[key])
        if not os.environ.get("GIDS_DIR"):
            raise ValueError(
                "Set OceanDisk config_dir or GIDS_DIR before initialization."
            )
        # Native code concatenates the directory and filename.
        os.environ["GIDS_DIR"] = os.path.join(os.environ["GIDS_DIR"], "")
        if "use_single_port" in config:
            if type(config["use_single_port"]) is not bool:
                raise ValueError("OceanDisk use_single_port must be a boolean.")
            os.environ["USE_SINGLE_PORT"] = "1" if config["use_single_port"] else "0"
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        config_path = gids_config_path(
            os.environ["GIDS_DIR"], device_index, os.environ.get("USE_SINGLE_PORT")
        )
        connection_count = read_connection_count(config_path)
        if "connection_indices" not in config:
            self.connection_indices = tuple(range(connection_count))
        if any(index >= connection_count for index in self.connection_indices):
            raise ValueError(
                f"OceanDisk connection_indices refer outside {connection_count} connections "
                f"in {config_path}."
            )

        self.connection_router = ConnectionRouter(config_path, self.connection_indices)
        with self._registration_lock, torch.cuda.device(self.device):
            if type(self)._initialized:
                raise RuntimeError(
                    "OceanDisk process-global state is already initialized; restart the "
                    "process rather than reinitializing registered GPU memory."
                )
            torch.ops.load_library(library)
            self._ops = torch.ops.my_ops
            self._get_stream = torch.cuda.Stream(device=self.device)
            self._put_stream = torch.cuda.Stream(device=self.device)
            self._output = torch.zeros(2, dtype=torch.int64, device=self.device)
            # Match the old controller's reusable EXISTS metadata buffers.
            self._exists_metadata_cpu = torch.empty(
                GET_METADATA_WIDTH, dtype=torch.int64, device="cpu", pin_memory=True
            )
            self._exists_metadata_np = self._exists_metadata_cpu.numpy()
            self._exists_metadata_gpu = torch.empty(
                GET_METADATA_WIDTH, dtype=torch.int64, device=self.device
            )
            addresses, lengths = zip(*self.regions)
            # Match the old controller's MR extension for direct padded I/O.
            # This extends registration only; it does not allocate trailing memory.
            lengths = [length + 512 for length in lengths]
            tensor = torch.tensor(
                [*addresses, *lengths], dtype=torch.int64, device=self.device
            )
            self._get_stream.wait_stream(torch.cuda.current_stream(self.device))
            # Even failed init may have registered memory: retain it and prohibit retry.
            type(self)._initialized = True
            type(self)._registered_tensors = list(buffers)
            with torch.cuda.stream(self._get_stream):
                try:
                    # Match od/nds_utils.py: stagger GIDS connection setup by GPU.
                    time.sleep(device_index * 2)
                    status = self._ops.kvcache_init(tensor, self._output)
                finally:
                    self._get_stream.synchronize()
            self._check_status(status, "init")
            self._put_stream.wait_stream(self._get_stream)

    def _run_io(self, operation, stream):
        with (
            self._torch.cuda.device(self.device),
            self._torch.cuda.stream(stream),
        ):
            self._ensure_open()
            return operation()

    @staticmethod
    def _check_status(status, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"OceanDisk {operation} returned status {status!r}.")

    def exists_many(self, keys: list[int], *, timings=None) -> list[bool]:
        with self._torch.cuda.device(self.device):
            self._ensure_open()
            started = time.perf_counter() if timings is not None else 0.0
            result = (
                self._exists_many(keys, timings=timings)
                if timings is not None
                else self._exists_many(keys)
            )
            if timings is not None:
                timings["exists_ms"] = (time.perf_counter() - started) * 1000
            return result

    def _prepare_exists_metadata(self, keys, connections):
        count = len(keys)
        width = 1 + 2 * count
        if width > self._exists_metadata_cpu.numel():
            raise ValueError(
                f"OceanDisk EXISTS metadata needs {width} int64 values; "
                f"capacity is {self._exists_metadata_cpu.numel()}."
            )
        values = self._exists_metadata_np
        values[0] = next(self._task_ids)
        values[1 : 1 + count] = keys
        values[1 + count : width] = connections
        gpu = self._exists_metadata_gpu[:width]
        gpu.copy_(self._exists_metadata_cpu[:width], non_blocking=True)
        return gpu

    def _exists_many(self, keys: list[int], *, timings=None) -> list[bool]:
        """Return a prefix-only bitmap from one native batch existence query.

        kvcache_exist returns the contiguous hit prefix in input order, not
        individual key statuses. Entries after the first miss remain False.
        """
        torch = self._torch
        # Metadata uses the querying thread's current stream.
        with torch.cuda.device(self.device):
            self._ensure_open()
            if timings is not None:
                timings.update(
                    exists_prepare_ms=0.0,
                    exists_route_ms=0.0,
                    exists_metadata_ms=0.0,
                    exists_native_ms=0.0,
                )
                prepare_start = time.perf_counter()
            if not keys:
                return []
            connections = [self.connection_router.select(key) for key in keys]
            if timings is not None:
                route_done = time.perf_counter()
                timings["exists_route_ms"] = (route_done - prepare_start) * 1000
            metadata = self._prepare_exists_metadata(keys, connections)
            if timings is not None:
                metadata_done = time.perf_counter()
                timings["exists_metadata_ms"] = (metadata_done - route_done) * 1000
                timings["exists_prepare_ms"] = (metadata_done - prepare_start) * 1000
                native_start = time.perf_counter()
            prefix_hits = self._ops.kvcache_exist(metadata, self._output)
            if timings is not None:
                timings["exists_native_ms"] = (
                    time.perf_counter() - native_start
                ) * 1000
            if not isinstance(prefix_hits, int) or not 0 <= prefix_hits <= len(keys):
                raise RuntimeError(
                    f"OceanDisk batch exist returned invalid prefix length "
                    f"{prefix_hits!r} for {len(keys)} keys."
                )
        return [True] * prefix_hits + [False] * (len(keys) - prefix_hits)


    def _build_mr_index(self):
        ordered = sorted(
            (base, length, mr) for mr, (base, length) in enumerate(self.regions)
        )
        regions = np.asarray(ordered, dtype=np.int64)
        self._mr_starts = regions[:, 0].copy()
        self._mr_ends = regions[:, 0] + regions[:, 1]
        self._mr_ids = regions[:, 2].copy()
        self._mr_disjoint = bool(np.all(self._mr_ends[:-1] <= self._mr_starts[1:]))

    def _pack_layer_batches(self, layers, *, output=None):
        if not layers:
            return []
        count = len(layers[0])
        if count == 0:
            return []
        if count > MAX_GIDS_RANGES:
            raise ValueError(
                f"OceanDisk IO count {count} exceeds native capacity {MAX_GIDS_RANGES}."
            )
        if count % 2 or any(len(layer) != count for layer in layers):
            raise ValueError("OceanDisk layers require equal, even IO counts.")
        if isinstance(layers[0], np.ndarray):
            return [self._pack_get_matrix(layers, output=output)]
        half = count // 2
        # Fix one connection per page for the whole operation, including both
        # KV/Indexer halves and every layer.
        selected_connections = {
            item.key: self.connection_router.select(item.key)
            for item in layers[0][:half]
        }
        batch = (
            np.empty((len(layers), 1 + 6 * count), dtype=np.int64)
            if output is None
            else output
        )
        for layer, ranges in enumerate(layers):
            row = batch[layer]
            row[0] = next(self._task_ids)
            mr, keys, offsets, remote, sizes, connections = row[1:].reshape(6, count)
            keys[:] = [item.key for item in ranges]
            if not np.array_equal(keys[:half], keys[half:]):
                raise ValueError("OceanDisk requires matching KV and Indexer halves.")
            ptrs = np.fromiter(
                (item.ptr for item in ranges), dtype=np.int64, count=count
            )
            sizes[:] = [item.size for item in ranges]
            remote[:] = [item.storage_offset for item in ranges]
            if np.any(remote < 0):
                raise ValueError("OceanDisk storage offsets must be nonnegative.")
            if self._mr_disjoint:
                indices = np.searchsorted(self._mr_starts, ptrs, side="right") - 1
                valid = (indices >= 0) & (sizes > 0)
                valid &= ptrs + sizes <= self._mr_ends[indices]
                if not np.all(valid):
                    bad = int(np.flatnonzero(~valid)[0])
                    resolve_mr(self.regions, int(ptrs[bad]), int(sizes[bad]))
                mr[:] = self._mr_ids[indices]
                offsets[:] = ptrs - self._mr_starts[indices]
            else:
                # Overlapping registrations retain first-registration semantics.
                resolved = [resolve_mr(self.regions, r.ptr, r.size) for r in ranges]
                mr[:], offsets[:] = np.asarray(resolved, dtype=np.int64).T
            connections[:half] = [selected_connections[int(key)] for key in keys[:half]]
            connections[half:] = connections[:half]
            # Validate logical ranges before applying the existing direct-I/O padding.
            sizes[:] = (sizes + 511) & ~511
        return [batch]

    def _pack_get_matrix(self, layers, *, output=None):
        """Fill all GET layers using array operations and one page-route pass."""
        count = len(layers[0])
        half = count // 2
        batch = (
            np.empty((len(layers), 1 + 6 * count), dtype=np.int64)
            if output is None else output
        )
        fields = batch[:, 1:].reshape(len(layers), 6, count)
        keys = layers[0][:, 0]
        if not np.array_equal(keys[:half], keys[half:]):
            raise ValueError("OceanDisk requires matching KV and Indexer halves.")
        # Fill strided views directly, avoiding a second full input matrix.
        for layer, values in enumerate(layers):
            if not np.array_equal(values[:, 0], keys):
                raise ValueError("OceanDisk GET keys must match across layers.")
            fields[layer, 2] = values[:, 1]  # Absolute addresses until MR resolution.
            fields[layer, 4] = values[:, 2]
            fields[layer, 3] = values[:, 3]
        batch[:, 0] = np.fromiter(
            (next(self._task_ids) for _ in layers), dtype=np.int64, count=len(layers)
        )
        fields[:, 1] = keys
        selected = {
            int(key): self.connection_router.select(int(key)) for key in keys[:half]
        }
        connections = np.fromiter(
            (selected[int(key)] for key in keys[:half]), dtype=np.int64, count=half
        )
        fields[:, 5, :half] = connections
        fields[:, 5, half:] = connections
        pointers, sizes, remote = fields[:, 2], fields[:, 4], fields[:, 3]
        if np.any(remote < 0):
            raise ValueError("OceanDisk storage offsets must be nonnegative.")
        if self._mr_disjoint:
            indices = np.searchsorted(self._mr_starts, pointers, side="right") - 1
            valid = (indices >= 0) & (sizes > 0)
            valid &= pointers + sizes <= self._mr_ends[indices]
            if not np.all(valid):
                layer, item = np.argwhere(~valid)[0]
                resolve_mr(self.regions, int(pointers[layer, item]), int(sizes[layer, item]))
            fields[:, 0] = self._mr_ids[indices]
            pointers -= self._mr_starts[indices]
        else:
            # Preserve first-registration semantics for overlapping MR regions.
            for layer in range(len(layers)):
                resolved = np.asarray([
                    resolve_mr(self.regions, int(ptr), int(size))
                    for ptr, size in zip(pointers[layer], sizes[layer], strict=True)
                ], dtype=np.int64)
                fields[layer, 0], pointers[layer] = resolved.T
        sizes[:] = (sizes + 511) & ~511
        return batch

    def reserve_get_metadata(self, num_layers):
        # Called during Linker initialization, before starting the load worker.
        rows = num_layers + 1
        capacity = rows * GET_METADATA_WIDTH
        self._get_metadata_cpu = self._torch.empty(
            capacity, dtype=self._torch.int64, device="cpu", pin_memory=True
        )
        self._get_metadata_gpu = self._torch.empty(
            capacity, dtype=self._torch.int64, device=self.device
        )
        # One fixed event per layer, just like the legacy controller. The
        # scheduler does not overlap GET batches; the worker confirms all
        # events before the next batch reuses metadata and records them again.
        with self._torch.cuda.device(self.device):
            self._get_layer_events = [
                self._torch.cuda.Event() for _ in range(num_layers)
            ]
            for event in self._get_layer_events:
                event.record(self._get_stream)

    def _get_metadata_views(self, num_layers, count):
        # Reuse only after all prior layer events complete, or failure drains GET.
        width = 1 + 6 * count
        required = num_layers * width
        if width > GET_METADATA_WIDTH or required > self._get_metadata_cpu.numel():
            raise ValueError(
                "OceanDisk GET metadata exceeds the fixed startup capacity "
                f"({GET_METADATA_WIDTH} int64 values per layer)."
            )
        # Flat storage gives contiguous rows even when a later task is smaller.
        cpu = self._get_metadata_cpu[:required].view(num_layers, width)
        gpu = self._get_metadata_gpu[:required].view(num_layers, width)
        return cpu, gpu

    def get_layers(
        self,
        layers,
        *,
        on_layer_ready=None,
        on_layer_submitted=None,
        defer_remaining=False,
    ):
        """Submit layer zero now; optionally return a worker continuation for the rest."""
        import od_utils

        torch = self._torch
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self._get_stream),
        ):
            self._ensure_open()
            output = None
            if layers and len(layers[0]) and len(layers[0]) <= MAX_GIDS_RANGES:
                cpu, gpu = self._get_metadata_views(len(layers), len(layers[0]))
                output = cpu.numpy()
            descriptors = self._pack_layer_batches(layers, output=output)
            submitted = []
            try:
                for batch in descriptors:
                    gpu.copy_(cpu, non_blocking=True)
                    events = self._get_layer_events[:len(layers)]
                    for event in events:
                        event.record(self._get_stream)
                    submitted.append((cpu, gpu, events))

                def submit_span(first, last):
                    # Slice both metadata tensors and use local layer indices:
                    # the native wrapper indexes CPU task IDs and GPU rows
                    # differently for a nonzero start layer.
                    for cpu, gpu, events in submitted:
                        self._run_io(
                            lambda: od_utils.batch_dispatch_get(
                                gpu[first:last],
                                gpu.shape[1],
                                cpu[first:last],
                                self._output,
                                [event.cuda_event for event in events[first:last]],
                                self._get_stream.stream_id,
                                self.device.index,
                                0,
                                last - first,
                                True,
                                False,
                            ),
                            self._get_stream,
                        )
                    if on_layer_submitted is not None:
                        for layer in range(first, last):
                            on_layer_submitted(
                                layer, tuple(events[layer] for _, _, events in submitted)
                            )

                def finish():
                    with torch.cuda.device(self.device), torch.cuda.stream(self._get_stream):
                        try:
                            if len(layers) > 1:
                                submit_span(1, len(layers))
                            # In direct-wait mode the consumer confirms each layer.
                            # The worker only needs the stream's final completion.
                            wait_layers = (
                                range(len(layers) - 1, len(layers)) if layers
                                and on_layer_submitted is not None and on_layer_ready is None
                                else range(len(layers))
                            )
                            for layer in wait_layers:
                                # A layer is usable only after every I/O batch containing
                                # its paired KV/Indexer fragments has completed.
                                for _, _, events in submitted:
                                    events[layer].synchronize()
                                if on_layer_ready is not None:
                                    on_layer_ready(layer)
                        except BaseException:
                            self._get_stream.synchronize()
                            raise

                if layers:
                    submit_span(0, 1)
                if defer_remaining:
                    return finish
                finish()
            except BaseException:
                self._get_stream.synchronize()
                raise

    def put_layers(self, layers) -> None:
        import od_utils

        torch = self._torch
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self._put_stream),
        ):
            self._ensure_open()
            descriptors = self._pack_layer_batches(layers)
            submitted = []
            try:
                for batch in descriptors:
                    cpu = torch.from_numpy(batch)
                    gpu = cpu.to(self.device)
                    lengths = torch.tensor(
                        [len(row) for row in batch], dtype=torch.int64, device="cpu"
                    )
                    indices = torch.tensor(
                        list(range(len(batch))), dtype=torch.int64, device="cpu"
                    )
                    # Materialize CUDA handles; the native wrapper records
                    # each event again after the corresponding layer PUT.
                    events = [torch.cuda.Event() for _ in layers]
                    for event in events:
                        event.record(self._put_stream)
                    submitted.append((cpu, gpu, lengths, indices, events))
                    self._run_io(
                        lambda: od_utils.batch_dispatch_put(
                            gpu,
                            lengths,
                            indices,
                            cpu,
                            self._output,
                            [event.cuda_event for event in events],
                            self._put_stream.stream_id,
                            self.device.index,
                            0,
                            len(layers),
                            True,
                            False,
                        ),
                        self._put_stream,
                    )
                for _, _, _, _, events in submitted:
                    if events:
                        events[-1].synchronize()
            except BaseException:
                self._put_stream.synchronize()
                raise

    def reserve_layerwise_put_metadata(self, num_layers):
        """Reserve the legacy controller's PUT capacity on the writer only."""
        capacity = PUT_METADATA_REQUESTS * num_layers * PUT_METADATA_ROW_WIDTH
        torch = self._torch
        self._layerwise_put_cpu = torch.empty(
            capacity, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._layerwise_put_gpu = torch.empty(
            capacity, dtype=torch.int64, device=self.device
        )
        self._layerwise_put_free = [(0, capacity)]
        self._layerwise_put_leases = {}

    def _acquire_layerwise_put_metadata(self, required):
        # Only the offload worker allocates and releases these regions.
        for index, (offset, size) in enumerate(self._layerwise_put_free):
            if size >= required:
                if size == required:
                    self._layerwise_put_free.pop(index)
                else:
                    self._layerwise_put_free[index] = (offset + required, size - required)
                return offset
        raise RuntimeError(
            f"OceanDisk PUT metadata arena exhausted: needs {required} int64 values."
        )

    def _release_layerwise_put_region(self, offset, size):
        regions = sorted([*self._layerwise_put_free, (offset, size)])
        merged = []
        for start, length in regions:
            if merged and merged[-1][0] + merged[-1][1] == start:
                previous, available = merged[-1]
                merged[-1] = (previous, available + length)
            else:
                merged.append((start, length))
        self._layerwise_put_free = merged

    def release_layerwise_put_metadata(self, prepared):
        """Called by the worker only after the session's finish event completes."""
        offset, size = self._layerwise_put_leases.pop(id(prepared))
        self._release_layerwise_put_region(offset, size)

    def prepare_layerwise_put(self, layers):
        """Prepare descriptors on the PUT stream and retain through completion."""
        torch = self._torch
        with torch.cuda.device(self.device), torch.cuda.stream(self._put_stream):
            self._ensure_open()
            width = 1 + 6 * len(layers[0])
            required = len(layers) * width
            offset = self._acquire_layerwise_put_metadata(required)
            try:
                cpu = self._layerwise_put_cpu[offset : offset + required].view(len(layers), width)
                gpu = self._layerwise_put_gpu[offset : offset + required].view(len(layers), width)
                self._pack_layer_batches(layers, output=cpu.numpy())
                gpu.copy_(cpu, non_blocking=True)
                lengths = torch.tensor([width], dtype=torch.int64, device="cpu")
                indices = torch.tensor([0], dtype=torch.int64, device="cpu")
                prepared = [
                    [(cpu[layer : layer + 1], gpu[layer : layer + 1], lengths, indices)]
                    for layer in range(len(layers))
                ]
                self._layerwise_put_leases[id(prepared)] = (offset, required)
                return prepared
            except BaseException:
                # An upload may already have been submitted; retain its region
                # if synchronization itself fails.
                self._put_stream.synchronize()
                self._release_layerwise_put_region(offset, required)
                raise

    def submit_put_layer(self, layer, prepared, ready_event):
        import od_utils

        torch = self._torch
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self._put_stream),
        ):
            self._ensure_open()
            # Wait before taking the PUT lock; do not hold up EXISTS for compute.
            ready_event.synchronize()
            for cpu, gpu, lengths, indices in prepared:
                self._run_io(
                    lambda: od_utils.batch_dispatch_put(
                        gpu,
                        lengths,
                        indices,
                        cpu,
                        self._output,
                        [],
                        self._put_stream.stream_id,
                        self.device.index,
                        0,
                        1,
                        False,
                        False,
                    ),
                    self._put_stream,
                )

    def record_save_event(self):
        torch = self._torch
        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self._put_stream),
        ):
            event = torch.cuda.Event()
            event.record(self._put_stream)
            return event

    @staticmethod
    def _unsupported_hicache() -> NotImplementedError:
        return NotImplementedError(
            "OceanDiskStorage currently supports UnifiedCacheLinker range I/O only; "
            "HiCache host-cache integration is not implemented."
        )

    # Abstract HiCache methods remain unsupported by the UC adapter.
    def get(self, key, target_location=None, target_sizes=None):
        raise self._unsupported_hicache()

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        raise self._unsupported_hicache()

    def set(self, key, value=None, target_location=None, target_sizes=None):
        raise self._unsupported_hicache()

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        raise self._unsupported_hicache()

    def exists(self, key):
        raise self._unsupported_hicache()

    # Override base no-ops to reject unsupported host-cache operations.
    def batch_get_v1(self, keys, host_indices, extra_info=None):
        raise self._unsupported_hicache()

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        raise self._unsupported_hicache()

    def register_mem_pool_host(self, mem_pool_host):
        raise self._unsupported_hicache()

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        raise self._unsupported_hicache()

    def clear(self):
        raise self._unsupported_hicache()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OceanDisk storage is closed.")

    def close(self) -> None:
        if self._closed:
            return
        # Linker stops and joins its producers before closing Storage.
        with self._torch.cuda.device(self.device):
            self._get_stream.synchronize()
            self._put_stream.synchronize()
            self._get_metadata_cpu = None
            self._get_metadata_gpu = None
            if hasattr(self, "_layerwise_put_cpu"):
                self._layerwise_put_cpu = None
                self._layerwise_put_gpu = None
                self._layerwise_put_leases.clear()
            self._closed = True
        # GIDS exposes no deregistration operation; retain registered buffers.
