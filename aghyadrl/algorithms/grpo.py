import asyncio
import torch
from jaxtyping import Int, Float
from beartype import beartype
from aghyadrl.inference import InferenceServer, InferenceArgs
from transformers import AutoTokenizer
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class GRPOArgs:
    mini_batch_size: int = 16
    group_size: int = 8

    max_prompt_len: int = 200
    max_response_len: int = 1000

    train_data: str = "data/gsm8k/train.parquet"
    test_data: Optional[str] = None


async def generate_batch(inference_server: InferenceServer, batch: Int[torch.Tensor, "batch_size max_prompt_len"], micro_bsz: int):
    tasks = [
        asyncio.create_task(inference_server.generate(prompt.tolist()))
        for prompt in batch
    ]

    to_return = []
    pending = tasks    
    while pending:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for done_task in done:
            to_return.append(await done_task.result().json())
            if len(to_return) == micro_bsz:
                yield to_return
                to_return = []
    if len(to_return) > 0:
        yield to_return


async def main(grpo_args: GRPOArgs, inference_server: InferenceServer):
    while True:
        try:
            await inference_server.check_health()
            await asyncio.sleep(1)
            print(f"health check passed")
            break
        except:
            await asyncio.sleep(1)
            print(f"waiting...")
            continue
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    test_input = "hello, world!"
    input_tokens = tokenizer(test_input, return_tensors="pt")["input_ids"]
    batch = input_tokens.expand(4, *input_tokens.shape[1:] )
    print(f"{batch.shape=}")

    async for micro_batch in generate_batch(inference_server, batch, 2):
        print(f"started mock micro_bsz training, {micro_batch}")
        await asyncio.sleep(2)
        print(f"finished mock micro_bsz training")


if __name__ == "__main__":
    model_id = "Qwen/Qwen3-0.6B-Base"
    inference_args = InferenceArgs(model_id=model_id)
    grpo_args = GRPOArgs()
    inference_server = InferenceServer(inference_args)
    mp.set_start_method("spawn")
    start_p = mp.Process(target=inference_server.start_server)
    print(f"before start server")
    start_p.start()
    print(f"after start server")
    asyncio.run(main(grpo_args, inference_server))
    print(f"Exiting main")

    start_p.join()