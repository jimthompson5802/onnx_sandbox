import asyncio
import time
import numpy as np
import functools

def run_in_executor(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, lambda: f(*args, **kwargs))
    return inner

@run_in_executor
def do_something(time_out):
    time.sleep(time_out)
    return -time_out


async def do_something2(time_out):
    await asyncio.sleep(time_out)
    return -time_out

async def do_something_wrapper(time_out):
    return await do_something(time_out)

def main():
    do_something(3)
    do_something(3)
    do_something(3)

async def main2():
    results = await asyncio.gather(
        do_something2(3),
        do_something2(3),
        do_something2(3)
    )
    print(results)

async def main3():
    to_do = [
        asyncio.create_task(do_something_wrapper(x)) for x in [3, 3, 3, 3, 3]
    ]
    results = [await t for t in asyncio.as_completed(to_do)]
    print(results)


if __name__ == '__main__':
    t0 = time.perf_counter()
    # main()
    asyncio.run(main3())
    print(f'duration {time.perf_counter() - t0:0.2f}')