# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F


class CellModel(nn.Module):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
    ) -> None:
        super().__init__()

        std = 1e-3

        self.__weights = nn.Linear(input_size, neuron_number, bias=False)
        self.__recurrent_weights = nn.Linear(
            neuron_number, neuron_number, bias=False
        )
        self.__biases = nn.Parameter(th.randn(1, neuron_number) * std)

        def __init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=std)

        self.apply(__init_weights)

    def forward(self, x_t: th.Tensor, input_t: th.Tensor) -> th.Tensor:
        # x_t : (batch, input_size)
        return F.mish(
            self.__recurrent_weights(x_t)
            + self.__weights(input_t)
            + self.__biases
        )


class LiquidCell(nn.Module):
    def __init__(
        self, neuron_number: int, input_size: int, unfolding_steps: int
    ) -> None:
        super().__init__()

        std = 1e-3

        self.__a = nn.Parameter(th.randn(1, neuron_number) * std)
        self.__tau = nn.Parameter(th.abs(th.randn(1, neuron_number)) * std)

        self.__f = CellModel(neuron_number, input_size)

        self.__unfolding_steps = unfolding_steps

    def forward(
        self, x_t: th.Tensor, input_t: th.Tensor, delta_t: th.Tensor
    ) -> th.Tensor:
        x_t_next = x_t
        delta_t = delta_t.unsqueeze(1) / self.__unfolding_steps

        for _ in range(self.__unfolding_steps):
            x_t_next = (
                x_t_next + delta_t * self.__f(x_t_next, input_t) * self.__a
            ) / (1 + delta_t * (1 / self.__tau + self.__f(x_t_next, input_t)))

        return x_t_next
