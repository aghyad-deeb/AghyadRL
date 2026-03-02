import torch
from jaxtyping import Int, Float
from beartype import beartype
from aghyadrl.inference import InferenceServer, InferenceArgs
from transformers import AutoTokenizer


if __name__ == "__main__":
    model_id = "Qwen/Qwen3-0.6B-Base"
    inference_args = InferenceArgs(model_id=model_id)
    inference_server = InferenceServer(inference_args)
    inference_server.start_server()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    test_input = "hello, world!"
    input_tokens = tokenizer(test_input)["input_ids"]
    generation = inference_server.generate(input_tokens)
    print(f"{generation=}")