from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from main import InferenceServer

if __name__ == "__main__":
    server_args = ServerArgs(
        model_path="Qwen/Qwen3-0.6B",
        host=InferenceServer.HOST,
        port=InferenceServer.PORT,
        dp_size=1,
        tp_size=1,
        enable_lora=True,
        max_lora_rank=32,
        lora_target_modules=["q_proj", "k_proj", "v_proj",  "o_proj", "gate_proj", "up_proj", "down_proj"],
        mem_fraction_static=0.85,
        trust_remote_code=True,
        max_running_requests=256,
        load_balance_method="auto",
        reasoning_parser="qwen3"
    )
    launch_server(server_args)
    
