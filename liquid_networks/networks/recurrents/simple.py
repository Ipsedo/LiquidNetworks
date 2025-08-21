from typing import Callable

import torch as th
from torch import nn

from ..abstract_recurent import AbstractLiquidRecurrent


class LiquidRecurrent(AbstractLiquidRecurrent[th.Tensor]):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
        unfolding_steps: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
        output_size: int,
    ) -> None:
        super().__init__(
            neuron_number, input_size, unfolding_steps, activation_function
        )

        self.__neuron_number = neuron_number
        self.__to_output = nn.Linear(neuron_number, output_size)

    def _get_first_x(self, batch_size: int) -> th.Tensor:
        return self.cell_activation_function(
            th.zeros(
                batch_size,
                self.__neuron_number,
                device=next(self.parameters()).device,
            ),
        )

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        return i

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        out = self.__to_output(out)
        return out

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return th.stack(outputs, 1)
