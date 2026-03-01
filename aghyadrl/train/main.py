import torch
from jaxtyping import Float, Int
from beartype import beartype
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


class Trainer:

    TORCH_DTYPE_MAP = dict(
        float32=torch.float32,
        bfloat16=torch.bfloat16
    )

    def __init__(self, training_args):
        self.training_args = training_args
        self.model = None

    def load_model(self):
        torch_dtype = self.TORCH_DTYPE_MAP[self.training_args.dtype]
        model = AutoModelForCausalLM.from_pretrained(
            self.training_args.model_id,
            dtype=torch_dtype,
        )

        lora_config = LoraConfig(
            lora_alpha=32, # per lora_without_regret
            lora_dropout=0,
            r=self.training_args.lora_rank,
            target_modules=self.training_args.lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none", # seems to be the standard
        )

        #TODO add option to load an existing lora
        model = get_peft_model(model, lora_config)
        self.model = model
    
    @beartype
    def forward_backward(
        self,
        inputs: Int[torch.Tensor, "num_inputs max_seq_len"],
        advantages: Float[torch.Tensor, "num_inputs"],
        old_logprobs: Float[torch.Tensor, "num_inputs max_seq_len"],
        loss_function,
    ):
        # inputs: (num_inputs, max_input_len)
        # advantages: (num_inputs)
        #~ Maybe model.train
        #~ do we need padding? 
        micro_bsz = self.training_args.micro_batch_size
        #~ this shouldn't stay like this probably cause we maybe have a foreward
        #~ backward on num_inputs not divisible by micro_bsz and it's ok
        assert inputs.shape[0] % micro_bsz == 0, f"forward-backward num_inputs is not divisible by micro batch size. {inputs.shape=}, {micro_bsz=}"
        assert inputs.shape[0] == advantages.shape[0], f"num_inputs in `inputs` != num_inputs in `advantages`. {inputs.shape=}, {advantages.shape=}"
        num_iterations = inputs.shape[0] // micro_bsz
        micro_batches: Int[torch.Tensor, "num_iterations micro_bsz max_seq_len"] = inputs.view(num_iterations, micro_bsz, *inputs.shape[1:])
        micro_batches_advantages: Float[torch.Tensor, "num_iterations micro_bsz"] = advantages.view(num_iterations, micro_bsz)
        for i in range(num_iterations):
            mb: Int[torch.Tensor, "micro_bsz max_seq_len"] = micro_batches[i]
            mb_advantages: Int[torch.Tensor, "micro_bsz"] = micro_batches_advantages[i]
            logprobs: Float[torch.Tensor, "micro_bsz max_seq_len vocab_size"] = self.model(mb).logits
            loss = loss_function(mb, logprobs, old_logprobs, mb_advantages)
            loss.backward()

        

    #~ for better compute usage, I can calculate gradients as soon as I recieve 
    #~ rollout, but only take step once I reach batch size (maybe take gardient
    #~ at specific micro batch size as opposed to the full thing)