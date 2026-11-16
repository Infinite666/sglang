#!/bin/sh
set -x

prefill_ip="141.111.65.132"
decode_ip="141.111.65.134"
prefill_port=30002
decode_port=30001
bootstrap_port=31145
router_port=12345

python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://${prefill_ip}:${prefill_port} ${bootstrap_port} \
  --decode http://${decode_ip}:${decode_port} \
  --host 0.0.0.0 \
  --port ${router_port} \
  --health-check-timeout-secs 1000 \
