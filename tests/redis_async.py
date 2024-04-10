import asyncio
from asyncio import coroutines

import redis.asyncio as redis

pool = redis.ConnectionPool(host='localhost', port=6379, db=0)


async def main():
    client = redis.Redis.from_pool(pool)


    await client.hset("pk1", mapping={
        "feature1": 0.565656,
        "feature2": 0.12312321,

    })

    co = [client.hmget(f"pk{i}", keys=["feature1", "feature2"]) for i in range(1, 11)]
    dd = await asyncio.gather(*co)
    print(dd)


if __name__ == '__main__':
    redis_settings = {
        "HOST": "localhost",
        "PORT": 6379,
        "db": 0,
        "USERNAME": None,
        "PASSWORD": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
