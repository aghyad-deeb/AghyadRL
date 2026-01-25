python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 \
    --port 30000 \
    --data-parallel-size 2 \
    --mem-fraction-static 0.85 \
    --trust-remote-code \
    --max-running-requests 256 \
    --load-balance-method shortest_queue \
    --reasoning-parser qwen3

