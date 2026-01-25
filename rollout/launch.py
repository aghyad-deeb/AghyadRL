import requests
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs

MAX_RUNNING_REQUESTS = 256
HOST = "127.0.0.1"
PORT = 8112
BASE_URL = f"http://{HOST}:{PORT}"

def check_health():
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    # print(response.json)
    return response.status_code == 200

def start_server(rollout_args):
    server_args = ServerArgs(
        model_path=rollout_args.model_id,
        host=HOST,
        port=PORT,
        dp_size=rollout_args.data_parallel_size,
        tp_size=rollout_args.tensor_parallel_size,
        mem_fraction_static=rollout_args.mem_fraction_static,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=rollout_args.lora_rank,
        lora_target_modules=rollout_args.lora_target_modules,
        max_running_requests=MAX_RUNNING_REQUESTS,
        load_balancing_method="auto",
    )
    launch_server(server_args)

# dp_size must be 1 for lora weight syncing
def load_lora_adapter(lora_path):
    check_health()
    unload_url = f"{BASE_URL}/unload_lora_adapter"
    response = requests.post(
        unload_url,
        json=dict(
            lora_name="only_lora",
        )
    )
    return response
    # load_url


# For development
if __name__ == "__main__":
    server_args = ServerArgs(
        model_path="Qwen/Qwen3-0.6B",
        host=HOST,
        port=PORT,
        dp_size=1,
        tp_size=1,
        enable_lora=True,
        max_lora_rank=16,
        lora_target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"],
        mem_fraction_static=0.85,
        trust_remote_code=True,
        max_running_requests=256,
        load_balance_method="auto",
        reasoning_parser="qwen3"
    )
    launch_server(server_args)