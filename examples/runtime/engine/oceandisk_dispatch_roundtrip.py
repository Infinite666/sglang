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
from statistics import median

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


def report_routing(label, rows):
    """Read the metadata tensor native consumes and print its conn column."""
    count = (rows.shape[1] - 1) // 6
    keys = rows[:, 1 + count : 1 + 2 * count]
    sizes = rows[:, 1 + 4 * count : 1 + 5 * count]
    conns = rows[:, 1 + 5 * count : 1 + 6 * count]
    per_layer = [tuple(row) for row in conns.tolist()]
    if len(set(per_layer)) != 1:
        raise RuntimeError(f"{label}: connection column differs across layers")
    per_conn_bytes = Counter()
    for _, conn, size in zip(
        keys[0].tolist(), conns[0].tolist(), sizes[0].tolist()
    ):
        per_conn_bytes[conn] += size
    pairs = list(zip(keys[0].tolist(), conns[0].tolist()))
    print(
        f"{label}: io_per_layer={count} "
        f"conns_per_layer={dict(sorted(Counter(conns[0].tolist()).items()))} "
        f"bytes_per_conn={dict(sorted(per_conn_bytes.items()))} "
        f"max_conn={max(conns[0].tolist(), default=-1)}"
    )
    print(
        f"{label}: key->conn sample="
        f"{pairs[:8]}{' ...' if len(pairs) > 8 else ''}"
    )


def dispatch(label, submit, events, stream, per_layer=False, quiet=False):
    for event in events:
        event.record(stream)
    started = time.perf_counter()
    submit([event.cuda_event for event in events])
    returned = time.perf_counter()
    marks = []
    for event in events:
        event.synchronize()
        marks.append((time.perf_counter() - returned) * 1000)
    finished = time.perf_counter()
    metrics = {
        "call_ms": (returned - started) * 1000,
        "wait_ms": (finished - returned) * 1000,
        "total_ms": (finished - started) * 1000,
        "marks": marks,
    }
    if not quiet:
        print(
            f"{label}: call_ms={metrics['call_ms']:.3f} "
            f"event_wait_ms={metrics['wait_ms']:.3f} "
            f"total_ms={metrics['total_ms']:.3f}"
        )
        if per_layer:
            # Events are recorded by native after each layer's I/O on the same
            # stream, so they complete in layer order and each sync is cumulative.
            previous = 0.0
            for layer, mark in enumerate(marks):
                print(
                    f"{label} layer={layer}: ready_ms={mark:.3f} "
                    f"delta_ms={mark - previous:.3f}"
                )
                previous = mark
    return metrics


def dispatch_split(label, submit_span, events, stream, layers, quiet=False):
    """Layer 0 alone, then the rest, then wait. Mirrors get_layers()."""
    for event in events:
        event.record(stream)
    started = time.perf_counter()
    submit_span(0, 1, [events[0].cuda_event])
    first_returned = time.perf_counter()
    if layers > 1:
        submit_span(1, layers, [event.cuda_event for event in events[1:]])
    rest_returned = time.perf_counter()
    # Query happens after the second submit returns, so True only means
    # "already complete when queried", not "complete before that return".
    layer0_done_at_query = events[0].query()
    marks = []
    for event in events:
        event.synchronize()
        marks.append((time.perf_counter() - rest_returned) * 1000)
    finished = time.perf_counter()
    metrics = {
        "layer0_call_ms": (first_returned - started) * 1000,
        "rest_call_ms": (rest_returned - first_returned) * 1000,
        "wait_ms": (finished - rest_returned) * 1000,
        "total_ms": (finished - started) * 1000,
        "marks": marks,
        "layer0_done_at_query": layer0_done_at_query,
    }
    if not quiet:
        print(
            f"{label}: layer0_call_ms={metrics['layer0_call_ms']:.3f} "
            f"rest_call_ms={metrics['rest_call_ms']:.3f} "
            f"event_wait_ms={metrics['wait_ms']:.3f} "
            f"total_ms={metrics['total_ms']:.3f} "
            f"layer0_done_at_query={layer0_done_at_query}"
        )
        previous = 0.0
        for layer, mark in enumerate(marks):
            print(
                f"{label} layer={layer}: ready_ms={mark:.3f} "
                f"delta_ms={mark - previous:.3f}"
            )
            previous = mark
    return metrics


def report_get(runs, split, layers):
    """Print medians across repeated GET measurements."""
    totals = sorted(run["total_ms"] for run in runs)
    waits = median(run["wait_ms"] for run in runs)
    print(
        f"GET: runs={len(runs)} total_ms median={median(totals):.3f} "
        f"min={totals[0]:.3f} max={totals[-1]:.3f} "
        f"event_wait_ms median={waits:.3f}"
    )
    if split:
        print(
            f"GET: layer0_call_ms median="
            f"{median(run['layer0_call_ms'] for run in runs):.3f} "
            f"rest_call_ms median="
            f"{median(run['rest_call_ms'] for run in runs):.3f} "
            f"layer0_done_at_query={runs[-1]['layer0_done_at_query']}"
        )
    medians = [median(run["marks"][layer] for run in runs) for layer in range(layers)]
    previous = 0.0
    for layer, mark in enumerate(medians):
        print(
            f"GET layer={layer}: ready_ms={mark:.3f} "
            f"delta_ms={mark - previous:.3f}"
        )
        previous = mark


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
    parser.add_argument(
        "--split-first-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "GET submits layer 0 alone, then the rest, like get_layers() "
            "(default: on; disable with --no-split-first-layer)"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="discarded GET iterations before measuring (default: 0)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="measured GET iterations, reported as medians (default: 1)",
    )
    args = parser.parse_args()
    layers, pages = args.layers, args.pages
    logical_sizes = (args.kv_bytes, args.index_bytes)
    if min(layers, pages, *logical_sizes) <= 0:
        parser.error("layers, pages and pool sizes must be positive")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0")
    sizes = tuple((size + 511) // 512 * 512 for size in logical_sizes)
    payloads = tuple(layers * pages * size for size in sizes)
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

    # Production keys are hashed page identities, i.e. uniformly spread over
    # the DPU ring. Consecutive integers almost always stay inside one ring
    # segment, so generate uniform 63-bit keys instead.
    def random_keys(count, taken):
        keys = []
        while len(keys) < count:
            key = 1 + secrets.randbelow((1 << 63) - 1)
            if key in taken:
                continue
            taken.add(key)
            keys.append(key)
        return keys

    taken = set()
    kv_keys = random_keys(pages, taken)
    index_keys = random_keys(pages, taken) if args.separate_pool_keys else kv_keys
    keys = kv_keys + index_keys
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
        f"kv_first_key={kv_keys[0]} index_first_key={index_keys[0]} "
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
        report_routing("PUT", put_cpu)
        report_routing("GET", get_cpu)
        put_gpu, get_gpu = put_cpu.to(device), get_cpu.to(device)
        width = put_cpu.shape[1]
        lengths = torch.full((layers,), width, dtype=torch.int64)
        indices = torch.arange(layers, dtype=torch.int64)
        tail = (stream.stream_id, args.device, 0, layers, True, False)

        def submit_span(first, last, event_handles):
            # Local slices plus start_layer=0: native indexes CPU task IDs and
            # GPU rows locally, so a nonzero start layer would misalign them.
            od_utils.batch_dispatch_get(
                get_gpu[first:last],
                width,
                get_cpu[first:last],
                output,
                event_handles,
                stream.stream_id,
                args.device,
                0,
                last - first,
                True,
                False,
            )

        dispatch(
            "PUT",
            lambda events: od_utils.batch_dispatch_put(
                put_gpu, lengths, indices, put_cpu, output, events, *tail
            ),
            [torch.cuda.Event() for _ in range(layers)],
            stream,
        )
        def run_get(quiet):
            events = [torch.cuda.Event() for _ in range(layers)]
            if args.split_first_layer:
                return dispatch_split(
                    "GET", submit_span, events, stream, layers, quiet=quiet
                )
            return dispatch(
                "GET",
                lambda event_handles: submit_span(0, layers, event_handles),
                events,
                stream,
                per_layer=True,
                quiet=quiet,
            )

        for _ in range(args.warmup):
            run_get(quiet=True)
        if args.repeat == 1:
            run_get(quiet=False)
        else:
            runs = [run_get(quiet=True) for _ in range(args.repeat)]
            report_get(runs, args.split_first_layer, layers)

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
