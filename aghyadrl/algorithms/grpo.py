import asyncio
import torch
from jaxtyping import Int, Float
from beartype import beartype
from aghyadrl.inference import InferenceServer, InferenceArgs
from transformers import AutoTokenizer
import multiprocessing as mp

async def main():
    while True:
        try:
            await inference_server.check_health()
            print(f"health check passed")
            break
        except:
            await asyncio.sleep(1)
            print(f"waiting...")
            continue
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    test_input = "hello, world!"
    input_tokens = tokenizer(test_input)["input_ids"]
    generation = inference_server.generate(input_tokens)
    print(f"{generation=}")

if __name__ == "__main__":
    model_id = "Qwen/Qwen3-0.6B-Base"
    inference_args = InferenceArgs(model_id=model_id)
    inference_server = InferenceServer(inference_args)
    mp.set_start_method("spawn")
    start_p = mp.Process(target=inference_server.start_server)
    print(f"before start server")
    start_p.start()
    print(f"after start server")
    asyncio.run(main())
    print(f"Exiting main")

    start_p.join()