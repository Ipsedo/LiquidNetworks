from typing import Callable

import torch as th
from torch import nn

from ..abstract_recurent import AbstractLiquidRecurrent
from ..factory import AbstractLiquidRecurrentFactory


class BfrbLiquidRecurrent(
    AbstractLiquidRecurrent[tuple[th.Tensor, th.Tensor]]
):
    def __init__(
        self,
        neuron_number: int,
        unfolding_steps: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
        dropout: float,
    ) -> None:
        channels = [
            (1, 8),
            (8, 16),
            (16, 32),
        ]

        ltc_input_size = self.nb_features + channels[-1][1] * self.nb_grids

        super().__init__(
            neuron_number,
            ltc_input_size,
            unfolding_steps,
            activation_function,
        )

        self.__grid_encoders = nn.ModuleList(
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Conv2d(
                            c_i, c_o, kernel_size=3, stride=1, padding=1
                        ),
                        nn.Mish(),
                        nn.Conv2d(
                            c_o, c_o, kernel_size=3, stride=2, padding=1
                        ),
                        nn.Mish(),
                    )
                    for c_i, c_o in channels
                ]
            )
            for _ in range(self.nb_grids)
        )

        self.__to_output = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(neuron_number, self.output_size)
        )

        self.__cls_token = nn.Parameter(th.zeros(1, 1, ltc_input_size))

    def _process_input(self, i: tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        grids, features = i

        b, t = grids.size()[:2]

        encoded_grids = th.cat(
            [
                th.unflatten(
                    self.__grid_encoders[i](
                        grids[:, :, i].flatten(0, 1).unsqueeze(1)
                    ).flatten(1, -1),
                    0,
                    (b, t),
                )
                for i in range(self.nb_grids)
            ],
            dim=-1,
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


class BfrbLiquidRecurrentFactory(
    AbstractLiquidRecurrentFactory[tuple[th.Tensor, th.Tensor]]
):
    def get_recurrent(
        self,
        neuron_number: int,
        unfolding_steps: int,
        act_fn: Callable[[th.Tensor], th.Tensor],
    ) -> AbstractLiquidRecurrent[tuple[th.Tensor, th.Tensor]]:
        return BfrbLiquidRecurrent(
            neuron_number,
            unfolding_steps,
            act_fn,
            self._get_config("dropout", float),
        )
