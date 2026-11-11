"""Offline JSONL inference with HBM and optional OceanDisk prefix-cache checks."""

import argparse
import json
import os
import time
from pathlib import Path


def build_prompt(tokenizer, data):
    question = (
        "阅读以下文字并用中文简短回答：\n\n"
        f"{data['context']}\n\n"
        "现在请基于上面的文章回答下面的问题，只告诉我答案，"
        "不要输出任何其他字词。\n\n"
        f"问题：{data['input']}\n回答："
    )
    messages = [
        {
            "role": "system",
            "content": (
                "先读问题，再根据下面的文章内容回答问题，"
                "不要进行分析，不要重复问题，用简短的语句给出答案。"
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_request(engine, prompt, label):
    print(f"\n{'=' * 20} {label} {'=' * 20}", flush=True)
    started = time.perf_counter()
    arrivals = []
    completion_tokens = 0
    output = None
    for output in engine.generate(
        prompt,
        sampling_params={
            "temperature": 0,
            "top_p": 0.95,
            "max_new_tokens": 16,
            "stream_interval": 1,
        },
        stream=True,
    ):
        received = time.perf_counter()
        count = output["meta_info"]["completion_tokens"]
        if count > completion_tokens:
            arrivals.append((received, count - completion_tokens))
            completion_tokens = count
    elapsed = time.perf_counter() - started
    if output is None:
        raise RuntimeError("Generation returned no output.")
    meta = output.get("meta_info", {})
    details = meta.get("cached_tokens_details") or {}
    # Measure Engine delivery, including scheduling, tokenization and IPC.
    # A final status-only message must not extend token-to-token intervals.
    ttft = (
        (arrivals[0][0] - started) * 1000 if arrivals and arrivals[0][1] == 1 else None
    )
    single_token_chunks = bool(arrivals) and all(count == 1 for _, count in arrivals)
    tbts = (
        [(right[0] - left[0]) * 1000 for left, right in zip(arrivals, arrivals[1:])]
        if single_token_chunks
        else []
    )
    print(f"Total latency: {elapsed:.3f}s (full response)", flush=True)
    print(
        f"TTFT:          {ttft:.3f} ms"
        if ttft is not None
        else "TTFT:          N/A (no individually delivered first token)",
        flush=True,
    )
    print(
        f"TBT mean:      {sum(tbts) / len(tbts):.3f} ms"
        if tbts
        else "TBT mean:      N/A (fewer than two tokens or merged token chunks)",
        flush=True,
    )
    if tbts:
        print(f"TBT each (ms): {[round(value, 3) for value in tbts]}", flush=True)
    elif arrivals and not single_token_chunks:
        print(f"Tokens/chunk:  {[count for _, count in arrivals]}", flush=True)
    print(f"Prompt tokens: {meta.get('prompt_tokens')}", flush=True)
    print(f"Output tokens: {meta.get('completion_tokens')}", flush=True)
    print(f"Cached tokens: {meta.get('cached_tokens')}", flush=True)
    print(f"Cache details: {details}", flush=True)
    print(f"Output:        {output['text']!r}", flush=True)
    return output


def test_prefix_cache(engine, prompt, use_linker):
    run_request(engine, prompt, "FIRST REQUEST - WARMUP")
    time.sleep(2)
    if not use_linker:
        second = run_request(engine, prompt, "SECOND REQUEST - HBM CACHE TEST")
        device_tokens = (second["meta_info"].get("cached_tokens_details") or {}).get(
            "device", 0
        )
        print(
            f"HBM Prefix Cache: {'HIT' if device_tokens > 0 else 'MISS / not confirmed'}",
            flush=True,
        )
        return

    flushed = engine.flush_cache()
    if not flushed.success:
        raise RuntimeError(f"Local cache flush failed: {flushed.message}")
    print("Local cache cleared; requesting the same prompt from OceanDisk.", flush=True)
    second = run_request(engine, prompt, "SECOND REQUEST - OCEANDISK LOAD-BACK TEST")
    # Direct Linker currently reports remote restoration through the host tier.
    details = second["meta_info"].get("cached_tokens_details") or {}
    if int(details.get("device", 0)) > 0:
        raise RuntimeError(
            "HBM cache hit remained after flush; external-only test failed."
        )
    remote_tokens = int(details.get("host", 0)) + int(details.get("storage", 0))
    if remote_tokens <= 0:
        raise RuntimeError(
            "OceanDisk cache MISS / load-back not confirmed after clearing local cache."
        )
    print(f"OceanDisk Prefix Cache: HIT ({remote_tokens} tokens)", flush=True)


def main():
    import sglang as sgl
    from sglang.srt.server_args import ServerArgs
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    ServerArgs.add_cli_args(parser)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=os.getenv(
            "DATASET_PATH", "/home/z00591349/dataset/multifieldqa_zh.jsonl"
        ),
    )
    args = parser.parse_args()
    print(f"SGLang path: {sgl.__file__}", flush=True)
    print(f"Model:       {args.model_path}", flush=True)
    print(f"Dataset:     {args.dataset_path}", flush=True)
    print(
        f"External Linker: enabled={args.enable_unified_cache_external_linker}; "
        f"backend={args.unified_cache_external_linker_backend}",
        flush=True,
    )
    with args.dataset_path.open(encoding="utf-8") as dataset:
        first_line = dataset.readline()
    if not first_line.strip():
        raise ValueError("Dataset is empty.")
    data = json.loads(first_line)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code, local_files_only=True
    )
    prompt = build_prompt(tokenizer, data)
    print(f"Prompt characters: {len(prompt)}", flush=True)
    args.enable_cache_report = True
    # Keep the final streamed result cumulative, as expected by the cache checks.
    args.incremental_streaming_output = False

    engine = sgl.Engine(server_args=ServerArgs.from_cli_args(args))
    try:
        test_prefix_cache(engine, prompt, args.enable_unified_cache_external_linker)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
