# %%

import torch
from jaxtyping import Int, Float
from beartype import beartype
from transformers import AutoTokenizer
from dataclasses import dataclass
from main import Trainer

# %%
@dataclass
class TrainingArgs:
    # model args
    model_id: str
    dtype: str

    # compute args
    micro_batch_size: int 

    # lora args
    lora_rank: int = 32
    lora_target_modules: str = "all-linear"

# %%
@beartype
def reinforce_loss(
    inputs: Int[torch.Tensor, "bsz max_seq_len"],
    logprobs: Float[torch.Tensor, "bsz max_seq_len vocab_size"],
    # we only need old_logprobs (and logprobs) for chosen tokens
    old_logprobs: Float[torch.Tensor, "bsz max_seq_len"], #~ assuming on-policy for now so not used
    #! Wait they still have the length normalization in the GRPO paper
    advantages: Float[torch.Tensor, "bsz"],
) -> Float[torch.Tensor, ""]:
    print(f"{inputs=}")
    print(f"{torch.gather(logprobs, -1, inputs.unsqueeze(-1)).squeeze(-1)=}")
    # print(f"{logprobs.tolist()=}")
    traj_logprob: Float[torch.Tensor, "bsz"] = (
        torch.gather(logprobs, -1, inputs.unsqueeze(-1)).squeeze(-1).sum(-1)
    ) / (torch.ones_like(inputs) * inputs.shape[-1])
    losses: Float[torch.Tensor, "bsz"] = traj_logprob * advantages
    # to turn from objective to los
    losses *= -1
    return losses.mean()
    
# %%
model_id = "Qwen/Qwen3-0.6B-Base"
args = TrainingArgs(
    model_id=model_id,
    dtype="bfloat16",
    micro_batch_size=1,
)
trainer = Trainer(training_args=args)
trainer.load_model()
tokenizer = AutoTokenizer.from_pretrained(model_id)
test_string = "hello, world!"
test_inputs = tokenizer(test_string, return_tensors="pt")["input_ids"]
print(f"{test_inputs.shape=}")
advantages = torch.ones(test_inputs.shape[0])
test_logprobs = trainer.model(test_inputs).logits
print(f"{test_logprobs=}")
old_logprobs = torch.gather(test_logprobs, -1, test_inputs.unsqueeze(-1)).squeeze(-1)

trainer.forward_backward(test_inputs, advantages, old_logprobs, vanilla_loss)

# %%
