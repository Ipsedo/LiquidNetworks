from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import torch as th

from .abstract_recurent import AbstractLiquidRecurrent


class MissingConfigException(Exception):
    def __init__(self, key: str) -> None:
        super().__init__(f"Missing config for '{key}'")


U = TypeVar("U")


class AbstractLiquidRecurrentFactory[T](ABC):
    def __init__(self, config: dict[str, str]) -> None:
        self.__config = config

    def _get_config(self, key: str, convert_fn: Callable[[str], U]) -> U:
        if key not in self.__config:
            raise MissingConfigException(key)
        return convert_fn(self.__config[key])

    @abstractmethod
    def get_recurrent(
        self,
        neuron_number: int,
        unfolding_steps: int,
        act_fn: Callable[[th.Tensor], th.Tensor],
    ) -> AbstractLiquidRecurrent[T]:
        pass
