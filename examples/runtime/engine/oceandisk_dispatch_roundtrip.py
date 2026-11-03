"""Standalone GIDS PUT/GET byte round trip using two layers and two pages.

Set OPEXTENSION and GIDS_DIR for the native installation. Run with Python
in an environment containing CUDA PyTorch and od_utils. This writes fresh
test keys to the configured GIDS namespace.
"""

import argparse
import os
import secrets
import time

import torch


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
    parser.add_argument("--conn", type=int, default=0, choices=range(8))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("OPEXTENSION", "GIDS_DIR"):
        if not os.environ.get(name):
            parser.error(f"set {name} for your GIDS installation")
    os.environ["GIDS_DIR"] = os.path.join(os.environ["GIDS_DIR"], "")

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.ops.load_library(os.environ["OPEXTENSION"])
    import od_utils

    layers, pages, size = 2, 2, 4096
    payload_bytes = layers * pages * size
    generator = torch.Generator().manual_seed(args.seed)
    sources = [
        torch.randint(0, 256, (payload_bytes,), generator=generator, dtype=torch.uint8)
        for _ in range(2)
    ]
    # One registered region per pool, with disjoint source and destination halves.
    buffers = [
        torch.full((2 * payload_bytes + 512,), 0xA5, device=device, dtype=torch.uint8)
        for _ in sources
    ]
    for buffer, source in zip(buffers, sources):
        buffer[:payload_bytes].copy_(source)

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

    base_key = (secrets.randbits(60) + 1) * pages
    keys = list(range(base_key, base_key + pages))
    print(f"keys={keys} conn={args.conn} seed={args.seed}")
    stream = torch.cuda.Stream(device=device)

    def metadata(destination_offset, first_task):
        rows = []
        for layer in range(layers):
            offsets = [
                destination_offset + (layer * pages + page) * size
                for page in range(pages)
            ]
            rows.append(
                [first_task + layer]
                + [0] * pages + [1] * pages
                + keys * 2
                + offsets * 2
                + [layer * size] * (2 * pages)
                + [size] * (2 * pages)
                + [args.conn] * (2 * pages)
            )
        return torch.tensor(rows, dtype=torch.int64)

    with torch.cuda.stream(stream):
        put_cpu = metadata(0, 1)
        get_cpu = metadata(payload_bytes, 1 + layers)
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

    for pool, (buffer, source) in enumerate(zip(buffers, sources)):
        actual = buffer[payload_bytes : 2 * payload_bytes].cpu()
        mismatch = (actual != source).nonzero(as_tuple=True)[0]
        if mismatch.numel():
            offset = mismatch[0].item()
            layer, page = divmod(offset // size, pages)
            raise RuntimeError(
                f"pool={pool} layer={layer} page={page} byte={offset % size}: "
                f"expected={source[offset].item()} actual={actual[offset].item()}"
            )
    print(f"PASS: verified {len(buffers) * payload_bytes} bytes")


if __name__ == "__main__":
    main()
