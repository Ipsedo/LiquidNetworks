from typing import Callable

import torch as th
from torch import nn
from torch.nn import functional as th_f

from ..abstract_recurent import AbstractLiquidRecurrent


class BfrbLiquidRecurrent(
    AbstractLiquidRecurrent[tuple[th.Tensor, th.Tensor]]
):
    def __init__(
        self,
        neuron_number: int,
        hidden_size: int,
        unfolding_steps: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
        output_size: int,
    ) -> None:
        self.__neuron_number = neuron_number

        nb_layers = 3
        factor = 2.0

        grid_input_size = 1
        self.__nb_grid = 5

        channels = [
            (
                int(grid_input_size * factor**i),
                int(grid_input_size * factor ** (i + 1)),
            )
            for i in range(nb_layers)
        ]

        nb_features = 12

        super().__init__(
            neuron_number, hidden_size, unfolding_steps, activation_function
        )

        self.__to_output = nn.Linear(neuron_number, output_size)

        self.__grid_encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(c_i, c_o, kernel_size=3, stride=1, padding=1),
                    nn.Mish(),
                    nn.Conv2d(c_o, c_o, kernel_size=3, stride=2, padding=1),
                    nn.Mish(),
                )
                for c_i, c_o in channels
            ]
        )

        self.__features_encoder = nn.Sequential(
            nn.Linear(nb_features, hidden_size),
            nn.Mish(),
            nn.LayerNorm(hidden_size),
        )

        self.__to_hidden_space = nn.Sequential(
            nn.Linear(
                hidden_size + channels[-1][1] * self.__nb_grid, hidden_size
            ),
            nn.Mish(),
            nn.LayerNorm(hidden_size),
        )

        self.__pooling = nn.Linear(neuron_number, 1)

    def _get_first_x(self, batch_size: int) -> th.Tensor:
        # pylint: disable=duplicate-code
        return self.cell_activation_function(
            th.zeros(
                batch_size,
                self.__neuron_number,
                device=next(self.parameters()).device,
            ),
        )

    def _process_input(self, i: tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        grids, features = i

        b, t = grids.size()[:2]

        encoded_grids = th.cat(
            [
                th.unflatten(
                    self.__grid_encoder(
                        grids[:, :, i].flatten(0, 1).unsqueeze(1)
                    )
                    .squeeze(-1)
                    .squeeze(-1),
                    0,
                    (b, t),
                )
                for i in range(self.__nb_grid)
            ],
            dim=2,
        )

        encoded_features = self.__features_encoder(features)

        encoded_input: th.Tensor = self.__to_hidden_space(
            th.cat([encoded_grids, encoded_features], dim=-1)
        )
        return encoded_input

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return out

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        stacked_outputs = th.stack(outputs, dim=1)
        out: th.Tensor = self.__to_output(
            th.sum(
                stacked_outputs
                * th_f.softmax(self.__pooling(stacked_outputs), dim=1),
                dim=1,
            )
        )
        return out
