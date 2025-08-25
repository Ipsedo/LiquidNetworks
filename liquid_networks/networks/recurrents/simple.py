from typing import Callable

import torch as th
from torch import nn

from ..abstract_recurent import AbstractLiquidRecurrent, AbstractLiquidRecurrentFactory


class LiquidRecurrent(AbstractLiquidRecurrent[th.Tensor]):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
        unfolding_steps: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
        delta_t: float,
        output_size: int,
    ) -> None:
        super().__init__(neuron_number, input_size, unfolding_steps, activation_function, delta_t)

        self.__to_output = nn.Linear(neuron_number, output_size)

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        return i

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        out = self.__to_output(out)
        return out

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return th.stack(outputs, 1)


class BaseLiquidRecurrentFactory[LtcConstructor: LiquidRecurrent](
    AbstractLiquidRecurrentFactory[th.Tensor]
):
    def __init__(self, config: dict[str, str], ltc_constructor: type[LtcConstructor]) -> None:
        super().__init__(config)
        self.__ltc_constructor = ltc_constructor

    def get_recurrent(
        self,
        neuron_number: int,
        unfolding_steps: int,
        act_fn: Callable[[th.Tensor], th.Tensor],
        delta_t: float,
    ) -> AbstractLiquidRecurrent[th.Tensor]:
        return self.__ltc_constructor(
            neuron_number,
            self._get_config("input_size", int),
            unfolding_steps,
            act_fn,
            delta_t,
            self._get_config("output_size", int),
        )


class LiquidRecurrentFactory(BaseLiquidRecurrentFactory[LiquidRecurrent]):
    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config, LiquidRecurrent)
