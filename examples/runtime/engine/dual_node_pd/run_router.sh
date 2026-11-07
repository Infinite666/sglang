#!/bin/sh
# 134 router, after run_decode.sh created the log directory.
set -x

# 141.111.65.* must not go through a proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://141.111.65.132:31002 31145 \
    --decode http://141.111.65.134:31001 \
    --host 0.0.0.0 \
    --port 31345 \
    --health-check-timeout-secs 1000 \
    2>&1 | tee -i /home/j00977581/pd_logs/latest/router.log
