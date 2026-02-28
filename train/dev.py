import torch
from jaxtyping import Int, Float
from beartype import beartype
from transformers import AutoTokenizer
from dataclasses import dataclass
from main import Trainer

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

@beartype
def vanilla_loss(
    inputs: Int[torch.Tensor, "bsz max_seq_len"],
    logprobs: Float[torch.Tensor, "bsz max_seq_len vocab_size"],
    advantages: Float[torch.Tensor, "bsz"],
):
    # outpus 
    traj_logprob: Float[torch.Tensor, "bsz"] = torch.gather(logprobs, -1, inputs.unsqueeze(-1)).squeeze(-1).sum(-1)
    losses: Float[torch.Tensor, "bsz"] = traj_logprob * advantages
    return losses

    

if __name__ == "__main__":
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
    test_output = trainer.model(test_inputs).logits
    print(f"{test_output=}")
    losses = vanilla_loss(test_inputs, test_output, advantages)
    print(f"{losses=}")
    # print(f"{test_output.shape=}")
    # loss_function = lambda x
    # trainer.forward_backward(test_inputs, advantages)