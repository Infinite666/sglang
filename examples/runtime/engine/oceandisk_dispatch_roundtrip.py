"""Standalone GIDS PUT/GET byte round trip with configurable layers, pages and pool sizes.

Set OPEXTENSION and GIDS_DIR for the native installation. Run with Python
in an environment containing CUDA PyTorch and od_utils. This writes fresh
test keys to the configured GIDS namespace.
"""

import argparse
import os
import secrets
import time
from bisect import bisect_left
from collections import Counter
from pathlib import Path

import torch


def gids_config_path(config_dir, device_index, use_single_port):
    single = use_single_port is None or use_single_port == "1"
    name = "gids_config.txt" if single else f"gids_config_{device_index}.txt"
    return Path(config_dir) / name


def load_connection_selector(config_path, selected=None):
    """Mirror od/nds_utils.py: hash keys onto DPU vnodes, then round-robin
    across that DPU's connections. A key must go to the DPU that owns it."""
    try:
        from xxhash3_py import xxh3_64_hexdigest
    except ImportError:
        from xxhash import xxh3_64_hexdigest

    groups = {}
    available = 0
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or not line.startswith("ip_config"):
            continue
        if selected is None or available in selected:
            fields = [part.strip() for part in line.split("=", 1)[1].split(",")]
            dpu = int(fields[5])
            groups.setdefault(dpu, []).append(available)
        available += 1
    if not groups:
        raise ValueError(f"No usable ip_config entries in {config_path}.")
    if selected is not None and any(index >= available for index in selected):
        raise ValueError(f"connection indices {selected} exceed {available} entries.")
    ring = sorted(
        (int(xxh3_64_hexdigest(f"DPU-{dpu}-V-{vnode}".encode("utf-8")), 16), dpu)
        for dpu in groups
        for vnode in range(50)
    )
    coordinates = [coordinate for coordinate, _ in ring]
    cursors = [0] * len(ring)

    def select(key):
        slot = bisect_left(coordinates, key % (1 << 64)) % len(ring)
        connections = groups[ring[slot][1]]
        connection = connections[cursors[slot] % len(connections)]
        cursors[slot] += 1
        return connection

    return select, available, sorted(groups)


def dispatch(label, submit, events, stream):
    for event in events:
        event.record(stream)
    started = time.perf_counter()
    submit([event.cuda_event for event in events])
    returned = time.perf_counter()
    for event in events:
        event.synchronize()
    finished = time.perf_counter()
    print(
        f"{label}: call_ms={(returned - started) * 1000:.3f} "
        f"event_wait_ms={(finished - returned) * 1000:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--conn",
        type=int,
        default=None,
        help="pin every I/O to one connection; default routes per key",
    )
    parser.add_argument(
        "--conns",
        type=str,
        default=None,
        help="comma-separated connection subset to route over; default all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--kv-bytes", type=int, default=4096)
    parser.add_argument("--index-bytes", type=int, default=4096)
    parser.add_argument("--separate-pool-keys", action="store_true")
    args = parser.parse_args()
    layers, pages = args.layers, args.pages
    logical_sizes = (args.kv_bytes, args.index_bytes)
    if min(layers, pages, *logical_sizes) <= 0:
        parser.error("layers, pages and pool sizes must be positive")
    sizes = tuple((size + 511) // 512 * 512 for size in logical_sizes)
    payloads = tuple(layers * pages * size for size in sizes)
    key_count = pages * (2 if args.separate_pool_keys else 1)
    if key_count >= 1 << 63:
        parser.error("too many keys for int64 metadata")
    for name in ("OPEXTENSION", "GIDS_DIR"):
        if not os.environ.get(name):
            parser.error(f"set {name} for your GIDS installation")
    os.environ["GIDS_DIR"] = os.path.join(os.environ["GIDS_DIR"], "")

    config_path = gids_config_path(
        os.environ["GIDS_DIR"], args.device, os.environ.get("USE_SINGLE_PORT")
    )
    selected = (
        [int(part) for part in args.conns.split(",") if part.strip()]
        if args.conns
        else None
    )
    if selected is not None and (not selected or len(set(selected)) != len(selected)):
        parser.error("--conns must list distinct connection indices")
    select_conn, available, dpu_ids = load_connection_selector(config_path, selected)
    if args.conn is not None and not 0 <= args.conn < available:
        parser.error(f"--conn must be in [0, {available}) from {config_path}")

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.ops.load_library(os.environ["OPEXTENSION"])
    import od_utils

    generator = torch.Generator().manual_seed(args.seed)
    sources = [
        torch.randint(0, 256, (payload_bytes,), generator=generator, dtype=torch.uint8)
        for payload_bytes in payloads
    ]
    # One registered region per pool, with disjoint source and destination halves.
    buffers = [
        torch.full((2 * payload_bytes + 512,), 0xA5, device=device, dtype=torch.uint8)
        for payload_bytes in payloads
    ]
    for buffer, source in zip(buffers, sources):
        buffer[:source.numel()].copy_(source)

    registration = torch.tensor(
        [b.data_ptr() for b in buffers] + [b.numel() for b in buffers],
        dtype=torch.int64,
        device=device,
    )
    output = torch.zeros(2, dtype=torch.int64, device=device)
    status = torch.ops.my_ops.kvcache_init(registration, output)
    torch.cuda.synchronize(device)
    if status != 0:
        raise RuntimeError(f"kvcache_init failed: {status}")

    # Leave room for all consecutive keys in signed int64 metadata.
    base_key = 1 + secrets.randbelow((1 << 63) - key_count)
    index_base = base_key + pages if args.separate_pool_keys else base_key
    keys = list(range(base_key, base_key + pages))
    keys.extend(range(index_base, index_base + pages))
    if args.conn is not None:
        conns = [args.conn] * len(keys)
    elif args.separate_pool_keys:
        conns = [select_conn(key) for key in keys]
    else:
        # Production fixes one connection per page for both KV/Indexer halves.
        conns = [select_conn(key) for key in keys[:pages]] * 2
    print(
        f"layers={layers} pages={pages} logical_pool_bytes={logical_sizes} "
        f"aligned_pool_bytes={sizes} bytes_per_direction={sum(payloads)} "
        f"kv_first_key={base_key} index_first_key={index_base} "
        f"seed={args.seed} config={config_path} connections={available} "
        f"dpus={dpu_ids} routed={dict(sorted(Counter(conns).items()))}"
    )
    stream = torch.cuda.Stream(device=device)

    def metadata(destination, first_task):
        rows = []
        for layer in range(layers):
            local_offsets = [
                (payload if destination else 0) + (layer * pages + page) * size
                for size, payload in zip(sizes, payloads)
                for page in range(pages)
            ]
            rows.append(
                [first_task + layer]
                + [0] * pages + [1] * pages
                + keys
                + local_offsets
                + [layer * size for size in sizes for _ in range(pages)]
                + [size for size in sizes for _ in range(pages)]
                + conns
            )
        return torch.tensor(rows, dtype=torch.int64)

    with torch.cuda.stream(stream):
        put_cpu = metadata(False, 1)
        get_cpu = metadata(True, 1 + layers)
        put_gpu, get_gpu = put_cpu.to(device), get_cpu.to(device)
        width = put_cpu.shape[1]
        lengths = torch.full((layers,), width, dtype=torch.int64)
        indices = torch.arange(layers, dtype=torch.int64)
        tail = (stream.stream_id, args.device, 0, layers, True, False)
        dispatch(
            "PUT",
            lambda events: od_utils.batch_dispatch_put(
                put_gpu, lengths, indices, put_cpu, output, events, *tail
            ),
            [torch.cuda.Event() for _ in range(layers)],
            stream,
        )
        dispatch(
            "GET",
            lambda events: od_utils.batch_dispatch_get(
                get_gpu, width, get_cpu, output, events, *tail
            ),
            [torch.cuda.Event() for _ in range(layers)],
            stream,
        )

    for pool, (buffer, source, size) in enumerate(zip(buffers, sources, sizes)):
        payload_bytes = source.numel()
        actual = buffer[payload_bytes : 2 * payload_bytes].cpu()
        mismatch = (actual != source).nonzero(as_tuple=True)[0]
        if mismatch.numel():
            offset = mismatch[0].item()
            layer, page = divmod(offset // size, pages)
            raise RuntimeError(
                f"pool={pool} layer={layer} page={page} byte={offset % size}: "
                f"expected={source[offset].item()} actual={actual[offset].item()}"
            )
    print(f"PASS: verified {sum(payloads)} bytes")


if __name__ == "__main__":
    main()
