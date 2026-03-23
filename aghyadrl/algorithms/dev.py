# %% 
import asyncio

async def s(t):
    await asyncio.sleep(t)
    return f"done with {t=}"

tasks = [asyncio.create_task(s(i * 2)) for i in range(5)]
pending = tasks
while pending:
    done, pending = await asyncio.wait(pending, return_when = "FIRST_COMPLETED")
    for done_task in done:
        print(f"{done_task.result()=}")
    print(f"{pending=}\n")
# %%
async def d(t):
    await asyncio.sleep(t)
    return f"d({t=}) returned"

async def it():
    for i in range(5):
        await asyncio.sleep(2)
        yield f"batch_{i}"

async for i in it():
    print(i)
    print(await d(1))
# %%

import asyncio

async def s(t):
    await asyncio.sleep(t)
    return f"done with {t=}"

tasks = [asyncio.create_task(s(i)) for i in range(3)]
(await asyncio.gather(*tasks))
# %%
