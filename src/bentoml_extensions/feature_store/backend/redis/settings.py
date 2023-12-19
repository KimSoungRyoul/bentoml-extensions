import os
from typing import Optional, List, Tuple

from pydantic_settings import BaseSettings


class DBSettings(BaseSettings):
    username: Optional[str] = None
    password: Optional[str] = None
    hosts: List[Tuple[str, int]] = [("127.0.0.1", 3000), ]
    namespace: str = "test"



if __name__ == '__main__':

    os.environ["PASSWORD"] = "111"
#    os.environ["HOSTS"] = "[('127.0.0.1', 3000), ]"

    settings = DBSettings()

    print(settings)