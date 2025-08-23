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
    ) -> None:
        channels = [
            (1, 8),
            (8, 16),
            (16, 32),
        ]

        nb_features = 12
        output_size = 18

        self.__nb_grids = 5

        super().__init__(
            neuron_number,
            nb_features + channels[-1][1] * self.__nb_grids,
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
            for _ in range(self.__nb_grids)
        )

        self.__to_output = nn.Linear(neuron_number, output_size)

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
                for i in range(self.__nb_grids)
            ],
            dim=-1,
        )

        return th.cat([encoded_grids, features], dim=-1)

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return out

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        out: th.Tensor = self.__to_output(
            th.mean(th.stack(outputs, dim=1), dim=1)
        )
        return out


class BfrbLiquidRecurrentFactory(
    AbstractLiquidRecurrentFactory[tuple[th.Tensor, th.Tensor]]
):
    def get_recurrent(
        self,
        neuron_number: int,
        unfolding_steps: int,
        act_fn: Callable[[th.Tensor], th.Tensor],
    ) -> AbstractLiquidRecurrent[tuple[th.Tensor, th.Tensor]]:
        return BfrbLiquidRecurrent(neuron_number, unfolding_steps, act_fn)
