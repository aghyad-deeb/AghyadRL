#%%
import torch

inp = torch.tensor(
    [
        [55, 72],
        [12, 30],
    ]
)

ind = torch.tensor([0, 1])

torch.gather(inp, dim=-1, ind.unsqueeze(-1))

# %%
