import asyncio
import torch
from jaxtyping import Int, Float
from beartype import beartype
from transformers import AutoTokenizer
import multiprocessing as mp
from dataclasses import dataclass, asdict, field
from typing import Optional
import pandas as pd

from aghyadrl.inference import InferenceServer, InferenceArgs
from aghyadrl.train import Trainer, TrainingArgs

@dataclass
class GRPOArgs:
    mini_bsz: int = 16
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


@dataclass
class MicroBatch:
    trajs_lst: list[list[dict]] = field(default_factory=list)
    prompts: list[list[int]] = field(default_factory=list)

    max_prompt_len: int = -1
    max_response_len: int = -1

def decode_tokens_individually(tensor):
    t = AutoTokenizer.from_pretrained("Qwen/Qwen3-8b")
    return t.batch_decode(tensor)

@beartype
def reinforce_loss(
    inputs: Int[torch.Tensor, "bsz max_seq_len"],
    logprobs: Float[torch.Tensor, "bsz max_seq_len vocab_size"],
    # we only need old_logprobs (and logprobs) for chosen tokens
    old_logprobs: Float[torch.Tensor, "bsz max_seq_len"], #~ assuming on-policy for now so not used
    #! Wait they still have the length normalization in the GRPO paper
    advantages: Float[torch.Tensor, "bsz"],
    response_masks: Int[torch.Tensor, "bsz max_seq_len"]
) -> Float[torch.Tensor, ""]:
    # print(f"{logprobs.tolist()=}")
    response_lengths: Int[torch.Tensor, "bsz"] = response_masks.sum(-1)
    #TODO I think I should sum after multiplying by response mask, but why is the
    #TODO logprobs for pad changing? is that normal?
    #! I'm suspicious that padding isn't supposed to use the pad_token_id, 
    #! the model's highest logprobs are for im_start.
    # It seems like this is fine, we'll multiple by response mask anyway

    #TODO there's also a hang I should investigate. edit: I think there isn't
    #! for debugging
    sampled_tokens = logprobs.argmax(-1)
    output_tokens_logprobs = torch.gather(logprobs, -1, inputs.unsqueeze(-1)).squeeze(-1)

    # We remove the last token as it was not sampled in inference
    output_tokens_logprobs = output_tokens_logprobs[:,:-1]
    # We shift the mask by one to match the output tokens
    output_tokens_mask = response_masks[:, 1:]
    # for i in range(inputs.shape[0]):
    #     print(f"{[(a, b, c, d) for a, b, c, d in zip(decode_tokens_individually(inputs[i]), response_masks[i], decode_tokens_individually(sampled_tokens[i]), output_tokens_mask[i])]=}")
        # seems to be working! Mask output when it should
    # We mask the output tokens that don't belong to the trajectory (prompt 
    # tokens + pad tokens output + last token not used for trajectory)
    output_tokens_logprobs *= output_tokens_mask
    traj_logprob: Float[torch.Tensor, "bsz"] = (
        output_tokens_logprobs.sum(-1)
    ) / response_lengths
    # I wrote out the math; after differentiation, this should give the sample
    # mean of the gradient of the grpo objective
    losses: Float[torch.Tensor, "bsz"] = traj_logprob * advantages
    # to turn from objective to loss
    losses *= -1
    mean = losses.mean()
    print(f"{mean=}")
    return mean

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
    max_response_len = 0
    trajs = []
    for ret in returns:
        traj = await ret.json()
        max_response_len = max(max_response_len, len(traj["output_ids"]))
        trajs.append(traj)
    return (trajs, prompt_ids, max_response_len)

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

    pending = tasks    
    while pending:
        # We grab the completed groups as soon as they're done since we don't 
        # need them to be in order, but keep each group together for advantage 
        # calculation
        # We store the current prompt_ids len and response_len so we can pad 
        # accordingly and forward_backward
        mb = MicroBatch()
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for done_task in done:
            trajs, prompt_ids, group_max_response_len = done_task.result()
            mb.trajs_lst.append(trajs)
            mb.prompts.append(prompt_ids)
            mb.max_prompt_len = max(mb.max_prompt_len, len(prompt_ids))
            mb.max_response_len = max(mb.max_response_len, group_max_response_len)
            if len(mb.trajs_lst) == micro_bsz:
                yield mb
                mb = MicroBatch()
    if len(mb.trajs_lst) > 0:
        yield mb.trajs_lst

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

def convert_micro_batch_to_data(micro_batch: list, pad_token_id: int):
    data = []
    for trajs, prompt_ids in zip(micro_batch.trajs_lst, micro_batch.prompts):
        group_data = []
        for traj in trajs:
            output_ids = traj["output_ids"]
            tokens = torch.tensor(prompt_ids + output_ids)
            pad_left = micro_batch.max_prompt_len - len(prompt_ids)
            pad_right = micro_batch.max_response_len - len(output_ids)
            tokens = torch.nn.functional.pad(
                tokens,
                (pad_left, pad_right), # applied to the last dimension
                "constant",
                pad_token_id,
            )
            assert tokens.shape[0] == micro_batch.max_response_len + micro_batch.max_prompt_len
            # Assumes single-turn conversations
            response_mask_lst = (
                [0] * micro_batch.max_prompt_len # prompt masking
                + [1] * len(output_ids) + [0] * (micro_batch.max_response_len - len(output_ids)) # masking padding tokens
            )
            assert len(response_mask_lst) == micro_batch.max_response_len + micro_batch.max_prompt_len
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
    
    training_args = TrainingArgs(
        model_id="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        micro_bsz=4,
    )
    trainer = Trainer(training_args)
    trainer.load_model()
    
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

    async for micro_batch in generate_batch(inference_server, batch, grpo_args.group_size, training_args.micro_bsz):
        # print(f"started mock micro_bsz training, {micro_batch}")
        data = convert_micro_batch_to_data(micro_batch, tokenizer.pad_token_id)
        data = compute_rewards(data)
        data = compute_advantages(data)
        # advantages = compute_advantages()
        # await asyncio.sleep(2)
        training_groups = torch.stack([
            torch.stack([datum.tokens for datum in group])
            for group in data
        ])
        response_masks = torch.stack([
            torch.stack([datum.response_mask for datum in group])
            for group in data
        ])
        advantage_groups = torch.stack([
            torch.tensor([datum.advantage for datum in group])
            for group in data
        ])
        #~ Need to make sure it's max_seq_len_compatibe
        training_micro_batch = training_groups.view((training_args.micro_bsz * grpo_args.group_size, -1))
        print(f"{training_micro_batch.size()=}")
        response_masks_micro_batch = response_masks.view((training_args.micro_bsz * grpo_args.group_size, -1))
        advantage_micro_batch = advantage_groups.view((training_args.micro_bsz * grpo_args.group_size))
        print(f"{advantage_micro_batch.size()=}")

        mock_logprobs = torch.ones_like(training_micro_batch)
        mock_old_logprobs = torch.ones(training_micro_batch.shape)
        trainer.forward_backward(
            inputs=training_micro_batch,
            advantages=advantage_micro_batch,
            old_logprobs=mock_old_logprobs,
            response_masks=response_masks_micro_batch,
            loss_function=reinforce_loss,
        )
        print(f"finished mock micro_bsz training")
    trainer.optim_step()


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