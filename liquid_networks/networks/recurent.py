# -*- coding: utf-8 -*-
from statistics import mean

import numpy as np
import torch as th
from torch import nn

from .cell import LiquidCell


class LiquidRecurrent(nn.Module):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
        unfolding_steps: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.__cell = LiquidCell(neuron_number, input_size, unfolding_steps)
        self.__neuron_number = neuron_number
        self.__to_output = nn.Sequential(
            nn.Linear(neuron_number, output_size), nn.Sigmoid()
        )

    def __get_first_x(self, batch_size: int) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        return th.tanh(
            th.randn(batch_size, self.__neuron_number, device=device)
        )

    def forward(self, i: th.Tensor, delta_t: th.Tensor) -> th.Tensor:
        b, _, steps = i.size()

        x_t = self.__get_first_x(b)

        results = []

        for t in range(steps):
            x_t = self.__cell(x_t, i[:, :, t], delta_t[:, t])
            results.append(self.__to_output(x_t))

        return th.stack(results, -1)

    def count_parameters(self) -> int:
        return sum(
            int(np.prod(p.size()))
            for p in self.parameters()
            if p.requires_grad
        )

    def grad_norm(self) -> float:
        return mean(
            float(p.grad.norm().item())
            for p in self.parameters()
            if p.grad is not None
        )
