# -*- coding: utf-8 -*-
import torch as th
from torch import nn


class MyLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(th.randn(1, in_features, out_features))
        self.bias = nn.Parameter(th.randn(1, out_features)) if bias else None

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = th.sum(self.weight * x.unsqueeze(-1), dim=1)
        if self.bias is not None:
            out += self.bias

        return out


class Model(nn.Module):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
    ) -> None:
        super().__init__()

        self.__weights = nn.Linear(input_size, neuron_number, bias=False)
        self.__recurrent_weights = nn.Linear(
            neuron_number, neuron_number, bias=False
        )
        self.__biases = nn.Parameter(th.randn(1, neuron_number) * 1e-3)

        def __init_weights(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, MyLinear)):
                nn.init.xavier_uniform_(module.weight, gain=1e-3)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=1e-3)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

        self.apply(__init_weights)

    def forward(self, x_t: th.Tensor, input_t: th.Tensor) -> th.Tensor:
        # x_t : (batch, input_size)
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
            x_t_next = (
                x_t_next + delta_t * self.__f(x_t_next, input_t) * self.__a
            ) / (1 + delta_t * (1 / self.__tau + self.__f(x_t_next, input_t)))

        return x_t_next
