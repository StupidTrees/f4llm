#
import time
import asyncio


# 定义异步函数
async def hello():
    await asyncio.sleep(10)
    print('Hello World:%s' % time.time())


if __name__ =='__main__':

    num = 5

    loop = asyncio.get_event_loop()
    tasks = [hello() for i in range(5)]

    for i in range(num):
        print(i, end=",")

    loop.run_until_complete(asyncio.wait(tasks))

