from abc import ABC
from typing import Callable, TypeVar


class MissingConfigException(Exception):
    def __init__(self, key: str) -> None:
        super().__init__(f"Missing config for '{key}'")


U = TypeVar("U")


class BaseFactory(ABC):
    def __init__(self, config: dict[str, str]) -> None:
        self.__config = config

    def _get_config(self, key: str, convert_fn: Callable[[str], U], default: U | None = None) -> U:
        if key in self.__config:
            return convert_fn(self.__config[key])
        if default is not None:
            return default

        raise MissingConfigException(key)
