import orjson
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

    data = {
        "pk1": orjson.dumps({"feature1": 0.565656, "feature2": 0.12312321}),
        "pk2": orjson.dumps({"feature1": 0.565656, "feature2": 0.12312321}),
        "pk3": orjson.dumps({"feature1": 0.565656, "feature2": 0.12312321}),
        "pk4": orjson.dumps({"feature1": 0.565656, "feature2": 0.12312321}),
        "pk5": orjson.dumps({"feature1": 0.565656, "feature2": 0.12312321}),
        "pk6": orjson.dumps({"feature1": 0.565656, "feature2": 0.12312321}),
        "pk7": orjson.dumps({"feature1": 0.565656, "feature2": 0.12312321}),

    }

    dd = r.mset(mapping=data)
    print(dd)

    print(r.mget(keys=["pk1", "pk2", "pk3", "pk4", "pk5", "pk6", "pk7", ]))

