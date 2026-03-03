import asyncio
import aiohttp
from typing import Optional
import requests
from dataclasses import dataclass, field
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs

async def async_get(url):
    session = aiohttp.ClientSession()
    resp = await session.request("GET", url)
    await session.close()
    return resp

async def async_post(url, json):
    session = aiohttp.ClientSession()
    resp = await session.request("POST", url, json=json)
    await session.close()
    return resp


@dataclass
class InferenceArgs:
    # model args
    model_id: str

    default_sampling_params: Optional[SamplingParams] = None
    #~ dtype: str

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1

    mem_fraction_static: float = 0.7

    # lora args
    lora_rank: int = 32
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj",  "o_proj", "gate_proj", "up_proj", "down_proj"])


class InferenceServer:

    MAX_RUNNING_REQUESTS = 256
    HOST = "127.0.0.1"
    PORT = 8112
    BASE_URL = f"http://{HOST}:{PORT}"
    FIXED_LORA_NAME = "only_lora"

    def __init__(self, inference_args: InferenceArgs):
        self.inference_args = inference_args
        # We override only parts specified in rollout config to keep model 
        # defaults if not specified (e.g. best temperature varies by model)
        self.default_sampling_params = inference_args.default_sampling_params
        # By having this None, we default to using no-loras
        self.curr_lora_name = None

    async def check_health(self):
        url = f"{self.BASE_URL}/health"
        response = await async_get(url)
        # print(response.json)
        return response.status == 200

    async def start_server(self):
        server_args = ServerArgs(
            model_path=self.inference_args.model_id,
            host=self.HOST,
            port=self.PORT,
            dp_size=self.inference_args.data_parallel_size,
            tp_size=self.inference_args.tensor_parallel_size,
            mem_fraction_static=self.inference_args.mem_fraction_static,
            trust_remote_code=True,
            enable_lora=True,
            max_lora_rank=self.inference_args.lora_rank,
            lora_target_modules=self.inference_args.lora_target_modules,
            max_running_requests=self.MAX_RUNNING_REQUESTS,
            #~ load_balancing_method="auto",
        )
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            None,
            launch_server(server_args)
        )

    # dp_size must be 1 for lora weight syncing
    async def load_lora_adapter(self, lora_path):
        assert await self.check_health()
        if self.curr_lora_name != None:
            unload_url = f"{self.BASE_URL}/unload_lora_adapter"
            unload_response = await async_post(
                unload_url,
                json=dict(
                    lora_name=self.FIXED_LORA_NAME,
                )
            )
            assert unload_response.status == 200, f"{unload_response.status=}\n{await unload_response.json()=}"

        load_url = f"{self.BASE_URL}/load_lora_adapter"
        load_response = await async_post(
            load_url,
            json=dict(
                lora_name=self.FIXED_LORA_NAME,
                lora_path=lora_path,
                pinned=True, # avoids unloading and reloading adapter
            )
        )
        assert load_response.status == 200, f"{load_response.status=}\n{await load_response.json()=}"
        self.curr_lora_name = self.FIXED_LORA_NAME
        return load_response

    async def generate(self, input_tokens, sampling_params=None):
        assert await self.check_health()
        if not sampling_params:
            sampling_params = self.default_sampling_params
        generate_json = dict(
            input_ids=input_tokens,
            sampling_params=sampling_params,
            lora_path=self.curr_lora_name,
        )
        url = f"{self.BASE_URL}/generate"
        response = await async_post(
            url,
            json=generate_json
        )
        return response
