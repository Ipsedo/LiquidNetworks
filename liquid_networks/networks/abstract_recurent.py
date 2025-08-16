from abc import ABC, abstractmethod
from statistics import mean
from typing import Callable

import numpy as np
import torch as th
from torch import nn

from .liquid_cell import LiquidCell


class AbstractLiquidRecurrent(ABC, nn.Module):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
        unfolding_steps: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__()

        self.__cell = LiquidCell(
            neuron_number, input_size, unfolding_steps, activation_function
        )

    @abstractmethod
    def _get_first_x(self, batch_size: int) -> th.Tensor:
        pass

    @abstractmethod
    def _process_input(self, i: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        pass

    def forward(self, i: th.Tensor, delta_t: th.Tensor) -> th.Tensor:
        b, _, _ = i.size()

        x_t = self._get_first_x(b)
        i = self._process_input(i)

        results = []

        for t in range(i.size(1)):
            x_t = self.__cell(x_t, i[:, t, :], delta_t[:, t])
            results.append(self._output_processing(x_t))

        return self._sequence_processing(results)

    def count_parameters(self) -> int:
        return sum(
            int(np.prod(p.size()))
            for p in self.parameters()
            if p.requires_grad
        )

    def grad_norm(self) -> float:
        return mean(
            float(p.grad.norm().item())
            for p in self.parameters()
            if p.grad is not None
        )

    def cell_activation_function(self, x: th.Tensor) -> th.Tensor:
        return self.__cell.activation_function(x)
