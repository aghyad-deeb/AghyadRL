import torch
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

def vanilla_loss(inputs, advantages):
    

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
    advantages = torch.ones_like(test_inputs)
    test_output = trainer.model(test_inputs)
    print(f"{test_output=}")
    # print(f"{test_output.shape=}")
    # loss_function = lambda x
    # trainer.foVkrward_backward(test_inputs, advantages)