import redis

if __name__ == '__main__':
    redis_settings = {
        "HOST": "localhost",
        "PORT": 6379,
        "db": 0,
        "USERNAME": None,
        "PASSWORD": None,
    }
    r = redis.Redis(host=redis_settings["HOST"],
                    port=redis_settings["PORT"], db=redis_settings["db"]
                    )

    dd = r.hmget("pk1", keys=["feature1","feature2"])
    print(dd)


    r.hset("pk1",mapping={
        "feature1": 0.565656,
        "feature2": 0.12312321,
    })

    print(r.hmget("pk1", keys=["feature1","feature2"]))


    print(r.hmget(keys=["pk1","pk2"]))