# -*- coding: utf-8 -*-
import torch as th
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
    ) -> None:
        super().__init__()

        self.__weights = nn.Linear(input_size, neuron_number)
        self.__recurrent_weights = nn.Linear(neuron_number, neuron_number)
        self.__biases = nn.Parameter(th.randn(1, neuron_number))

    def forward(self, x_t: th.Tensor, input_t: th.Tensor) -> th.Tensor:
        # x_t : (batch, input_size, time_steps)
        return th.tanh(
            self.__recurrent_weights(x_t)
            + self.__weights(input_t)
            + self.__biases
        )


class LiquidCell(nn.Module):
    def __init__(
        self, neuron_number: int, input_size: int, unfolding_steps: int
    ) -> None:
        super().__init__()

        self.__a = nn.Parameter(th.randn(1, neuron_number))
        self.__tau = nn.Parameter(th.randn(1, neuron_number))

        self.__f = Model(neuron_number, input_size)

        self.__unfolding_steps = unfolding_steps

    def forward(
        self, x_t: th.Tensor, input_t: th.Tensor, delta_t: th.Tensor
    ) -> th.Tensor:
        x_t_next = x_t
        delta_t = delta_t.unsqueeze(1) / self.__unfolding_steps

        for _ in range(self.__unfolding_steps):
            out = self.__f(x_t_next, input_t)
            x_t_next = (x_t_next + delta_t * out * self.__a) / (
                1 + delta_t * (1 / self.__tau * out)
            )

        return x_t_next
