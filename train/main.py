import torch
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
    
    def forward_backward(self, inputs: torch.Tensor, advantages: torch.Tensor, loss_function):
        # inputs: (num_inputs, max_input_len)
        # advantages: (num_inputs)
        #~ Maybe model.train
        #~ do we need padding? 
        micro_bsz = self.training_args.micro_batch_size
        assert inputs.shape[0] % micro_bsz == 0, f"forward-backward num_inputs is not divisible by micro batch size. {inputs.shape=}, {micro_bsz=}"
        assert inputs.shape[0] == advantages.shape[0], f"num_inputs in `inputs` != num_inputs in `advantages`. {inputs.shape=}, {advantages.shape=}"
        num_iterations = inputs.shape[0] / micro_bsz
        micro_batches = inputs.view(num_iterations, micro_bsz, *inputs.shape[:2])
        micro_batches_advantages = advantages.view(num_iterations, micro_bsz)
        for i in range(num_iterations):
            mb = micro_batches[i]
            mb_advantages = micro_batches_advantages[i]
            outputs = self.model(mb)
            loss = loss_function(outputs, mb_advantages)
            loss.backward()

        

    # for better compute usage, I can calculate gradients as soon as I recieve 
    # rollout, but only take step once I reach batch size (maybe take gardient
    # at specific micro batch size as opposed to the full thing)