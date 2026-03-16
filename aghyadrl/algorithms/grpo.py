import asyncio
import torch
from jaxtyping import Int, Float
from beartype import beartype
from aghyadrl.inference import InferenceServer, InferenceArgs
from transformers import AutoTokenizer
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd

@dataclass
class GRPOArgs:
    mini_bsz: int = 16
    micro_bsz: int = 4
    group_size: int = 8

    max_prompt_len: int = 200
    max_response_len: int = 1000

    train_data: str = "data/gsm8k/train.parquet"
    test_data: Optional[str] = None


@dataclass
class Datum:
    tokens: Int[torch.Tensor, "max_seq_len"]
    response_mask: Int[torch.Tensor, "max_seq_len"]
    response_text: str
    
    reward: Optional[float] = None
    advantage: Optional[float] = None


async def generate_group(
    inference_server: InferenceServer,
    prompt_ids: list[int],
    group_size: int
):
    tasks = [
        asyncio.create_task(inference_server.generate(prompt_ids))
        for _ in range(group_size)
    ]
    returns =  await asyncio.gather(*tasks)
    jsons = [await ret.json() for ret in returns]
    return (jsons, prompt_ids)

async def generate_batch(
    inference_server: InferenceServer,
    batch: Int[torch.Tensor, "batch_size max_prompt_len"],
    group_size: int,
    micro_bsz: int
):
    tasks = [
        asyncio.create_task(
            generate_group(inference_server, prompt_tensor.tolist(), group_size)
        )
        for prompt_tensor in batch
    ]

    to_return = []
    pending = tasks    
    while pending:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for done_task in done:
            to_return.append(done_task.result())
            if len(to_return) == micro_bsz:
                yield to_return
                to_return = []
    if len(to_return) > 0:
        yield to_return

def compute_reward(text):
    import random
    return random.random()

def compute_rewards(data: list):
    ndata = []
    for group in data:
        ngroup = []
        for datum in group:
            params = {
                **asdict(datum),
                "reward": compute_reward(datum.response_text),
            }
            ndatum = Datum(**params)
            ngroup.append(ndatum)
            #! for debugging
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            print(f"{tokenizer.decode(datum.tokens)=}")
        ndata.append(ngroup)
    return ndata

def compute_advantages(data: list):
    ndata = []
    for group in data:
        rewards = [datum.reward for datum in group]
        assert len(rewards) > 0
        avg_reward = sum(rewards) / len(rewards)
        ngroup = []
        for datum in group:
            # We don't divide by std deviation to avoid exaggerating small 
            # small rewards (like length penalties)
            advantage = datum.reward - avg_reward
            params = {
                **asdict(datum),
                "advantage": advantage,
            }
            ndatum = Datum(**params)
            ngroup.append(ndatum)
        ndata.append(ngroup)
    return ndata

def convert_micro_batch_to_data(micro_batch: list):
    data = []
    for group in micro_batch:
        trajs, prompt_ids = group
        group_data = []
        for traj in trajs:
            output_ids = traj["output_ids"]
            tokens = torch.tensor(prompt_ids + output_ids)
            # Assumes single-turn conversations
            response_mask_lst = [0] * len(prompt_ids) + [1] * len(output_ids)
            response_mask =  torch.tensor(response_mask_lst)
            datum = Datum(
                tokens=tokens,
                response_mask=response_mask,
                response_text=traj["text"],
            )
            group_data.append(datum)
        data.append(group_data)
    return data


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
    test_input = [
        {
            "role": "system",
            "content": "You're a helpful assistant",
        },
        {
            "role": "user",
            "content": "hello, world!",
        },
    ]
        
    input_tokens = tokenizer.apply_chat_template(test_input, return_tensors="pt", add_generation_prompt=True)
    batch = input_tokens.expand(4, *input_tokens.shape[1:])
    print(f"{batch.shape=}")

    async for micro_batch in generate_batch(inference_server, batch, grpo_args.group_size, grpo_args.mini_bsz):
        # print(f"started mock micro_bsz training, {micro_batch}")
        data = convert_micro_batch_to_data(micro_batch)
        data = compute_rewards(data)
        data = compute_advantages(data)
        print(f"{data=}")
        # advantages = compute_advantages()
        await asyncio.sleep(2)
        print(f"finished mock micro_bsz training")


if __name__ == "__main__":
    model_id = "Qwen/Qwen3-0.6B"
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