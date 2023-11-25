from asyncio import gather, run


async def foo(id):
    for n in range(100):
        # await sleep(0)
        print(id, n)


async def main():
    await gather(*[foo(i) for i in range(10)])


run(main())
