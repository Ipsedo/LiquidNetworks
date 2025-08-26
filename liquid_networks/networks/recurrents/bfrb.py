from typing import Callable

import torch as th
from torch import nn

from ..abstract_recurent import AbstractLiquidRecurrent, AbstractLiquidRecurrentFactory
from ..function import ActFnModule

# With grids


class BfrbLiquidRecurrent(AbstractLiquidRecurrent[tuple[th.Tensor, th.Tensor]]):
    def __init__(
        self,
        neuron_number: int,
        unfolding_steps: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
        delta_t: float,
        dropout: float,
    ) -> None:
        channels = [
            (self.nb_grids, 16),
            (16, 32),
            (32, 64),
        ]

        ltc_input_size = self.nb_features + channels[-1][1]

        super().__init__(
            neuron_number, ltc_input_size, unfolding_steps, activation_function, delta_t
        )

        self.__grid_encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(c_i, c_o, kernel_size=3, stride=2, padding=1),
                    ActFnModule(self._activation_function),
                    nn.GroupNorm(c_o, c_o, affine=True),
                )
                for c_i, c_o in channels
            ]
        )

        self.__to_output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(neuron_number, self.output_size),
        )

        self.__cls_token = nn.Parameter(th.zeros(1, 1, ltc_input_size))

    def _process_input(self, i: tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        grids, features = i

        b, t = grids.size()[:2]

        encoded_grids = th.unflatten(
            self.__grid_encoder(grids.flatten(0, 1)).flatten(1, -1), 0, (b, t)
        )

        return th.cat(
            [
                th.cat([encoded_grids, features], dim=-1),
                self.__cls_token.repeat(b, 1, 1),
            ],
            dim=1,
        )

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        logits: th.Tensor = self.__to_output(out)
        return logits

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return outputs[-1]

    @property
    def nb_grids(self) -> int:
        return 5

    @property
    def grid_size(self) -> tuple[int, int]:
        return 8, 8

    @property
    def nb_features(self) -> int:
        return 12

    @property
    def output_size(self) -> int:
        return 18


class BfrbLiquidRecurrentFactory(AbstractLiquidRecurrentFactory[tuple[th.Tensor, th.Tensor]]):
    def get_recurrent(
        self,
        neuron_number: int,
        unfolding_steps: int,
        act_fn: Callable[[th.Tensor], th.Tensor],
        delta_t: float,
    ) -> AbstractLiquidRecurrent[tuple[th.Tensor, th.Tensor]]:
        return BfrbLiquidRecurrent(
            neuron_number, unfolding_steps, act_fn, delta_t, self._get_config("dropout", float)
        )


# Without grids


class BfrbFeaturesOnlyLiquidRecurrent(AbstractLiquidRecurrent[th.Tensor]):
    def __init__(
        self,
        neuron_number: int,
        unfolding_steps: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
        delta_t: float,
        dropout: float,
    ) -> None:
        super().__init__(
            neuron_number,
            self.nb_features,
            unfolding_steps,
            activation_function,
            delta_t,
        )

        self.__to_output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(neuron_number, self.output_size),
        )

        self.__cls_token = nn.Parameter(th.randn(1, 1, self.nb_features))

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        logits: th.Tensor = self.__to_output(out)
        return logits

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        return th.cat([i, self.__cls_token.repeat(i.size(0), 1, 1)], dim=1)

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return outputs[-1]

    @property
    def nb_features(self) -> int:
        return 12

    @property
    def output_size(self) -> int:
        return 18


class BfrbFeaturesOnlyLiquidRecurrentFactory(AbstractLiquidRecurrentFactory[th.Tensor]):
    def get_recurrent(
        self,
        neuron_number: int,
        unfolding_steps: int,
        act_fn: Callable[[th.Tensor], th.Tensor],
        delta_t: float,
    ) -> AbstractLiquidRecurrent[th.Tensor]:
        return BfrbFeaturesOnlyLiquidRecurrent(
            neuron_number, unfolding_steps, act_fn, delta_t, self._get_config("dropout", float)
        )
