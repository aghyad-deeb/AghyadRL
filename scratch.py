# %%
from rollout.launch import check_health

# %%
resp = check_health()
# %%
resp.status_code == 200
# %%
from rollout.launch import load_lora_adapter

# %%
resp = load_lora_adapter("blah")
# %%
resp.json()
# %%
resp.status_code

# %%
import aiohttp 

async def async_get(url):
    session = aiohttp.ClientSession()
    resp = await session.request("GET", url)
    await session.close()
    return resp

resp = await async_get("http://localhost:8112/health")
# %%
resp.content

# %%
async def async_post(url, json):
    session = aiohttp.ClientSession()
    resp = await session.request("POST", url, json=json)
    await session.close()
    return resp


# %%
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6b")
messages = [
    {
        "role":"user", 
        "content": "ما عاصمة الصين؟"
    }
]
ids = t.apply_chat_template(messages, add_generation_prompt=True)
sampling_params = dict(max_new_tokens=3000)
# ids = t(text)
resp = await async_post("http://localhost:8112/generate", json=dict(input_ids=[ids, ids], sampling_params=sampling_params, lora_path=[None, infs.FIXED_LORA_NAME]))
# await resp.json()
# %%
await resp.json()

# %%
await resp.json()
# %%
t.decode(ids, skip_special_tokens=False)
# %%
from inference.launch import InferenceServer
from dataclasses import dataclass, field

@dataclass
class RolloutArgs:
    blah: int = 0
    default_sampling_params: dict = field(default_factory=dict)

rg = RolloutArgs()
infs = InferenceServer(rg)
# %%

resp = await infs.load_lora_adapter("YoussefHosni/Qwen3-0.6b-2-Token-arabic-LoRA-finetuned")
await resp.json()
# %%
