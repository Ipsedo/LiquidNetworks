from abc import ABC, abstractmethod
from statistics import mean
from typing import Callable

import numpy as np
import torch as th
from torch import nn

from .liquid_cell import LiquidCell


class AbstractLiquidRecurrent[T](ABC, nn.Module):
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

        self.__neuron_number = neuron_number

    def _get_first_x(self, batch_size: int) -> th.Tensor:
        return self.__cell.activation_function(
            th.zeros(
                batch_size,
                self.__neuron_number,
                device=next(self.parameters()).device,
            ),
        )

    @abstractmethod
    def _process_input(self, i: T) -> th.Tensor:
        pass

    @abstractmethod
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        pass

    def forward(self, i: T, delta_t: th.Tensor) -> th.Tensor:
        i_encoded = self._process_input(i)
        x_t = self._get_first_x(i_encoded.size(0))

        assert (
            len(i_encoded.size()) == 3
        ), "Processed input needs to have 3 dimensions (Batch, Time, Features)"

        results = []

        for t in range(i_encoded.size(1)):
            x_t = self.__cell(x_t, i_encoded[:, t, :], delta_t[:, t])
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
