from typing import Callable

import torch as th
from torch import nn


class ActFnModule(nn.Module):
    def __init__(self, activation: Callable[[th.Tensor], th.Tensor]) -> None:
        super().__init__()
        self.__activation = activation

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__activation(x)
