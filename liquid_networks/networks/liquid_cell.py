from typing import Callable

import torch as th
from torch import nn
from torch.nn import functional as th_f


class CellModel(nn.Module):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__()

        std = 1e-1

        def __init_weights(w: th.Tensor) -> None:
            th.nn.init.normal_(w, 0.0, std)

        self.__weights = nn.Linear(input_size, neuron_number, bias=False)
        self.__recurrent_weights = nn.Linear(neuron_number, neuron_number, bias=False)
        self.__biases = nn.Parameter(th.randn(1, neuron_number))

        self.__activation_function = activation_function

        __init_weights(self.__biases)
        __init_weights(self.__weights.weight)
        __init_weights(self.__recurrent_weights.weight)

    def forward(self, x_t: th.Tensor, input_t: th.Tensor) -> th.Tensor:
        # x_t: (batch, input_size)
        return self.__activation_function(
            self.__recurrent_weights(x_t) + self.__weights(input_t) + self.__biases
        )

    @property
    def activation_function(self) -> Callable[[th.Tensor], th.Tensor]:
        return self.__activation_function


class LiquidCell(nn.Module):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
        unfolding_steps: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__()

        self.__a = nn.Parameter(th.rand(1, neuron_number) - 0.5)

        self.__tau_min = 1e-2
        self.__tau_raw = nn.Parameter(th.rand(1, neuron_number) * 2.0 + 0.5)

        self.__f = CellModel(neuron_number, input_size, activation_function)

        self.__unfolding_steps = unfolding_steps

    @property
    def __tau(self) -> th.Tensor:
        # pylint: disable=not-callable
        return th_f.softplus(self.__tau_raw) + self.__tau_min

    def forward(self, x_t: th.Tensor, input_t: th.Tensor, delta_t: float) -> th.Tensor:
        x_t_next = x_t
        delta_t = delta_t / self.__unfolding_steps

        for _ in range(self.__unfolding_steps):
            f = self.__f(x_t_next, input_t)
            x_t_next = (x_t_next + delta_t * f * self.__a) / (1 + delta_t * (1 / self.__tau + f))

        return x_t_next

    @property
    def activation_function(self) -> Callable[[th.Tensor], th.Tensor]:
        return self.__f.activation_function
