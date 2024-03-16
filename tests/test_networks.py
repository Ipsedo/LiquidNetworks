# -*- coding: utf-8 -*-
from typing import Type

import pytest
import torch as th

from liquid_networks.networks.cell import CellModel, LiquidCell
from liquid_networks.networks.recurent import (
    LiquidRecurrent,
    LiquidRecurrentBrainActivity,
    LiquidRecurrentLast,
    LiquidRecurrentReg,
)


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("neuron_number", [2, 4])
@pytest.mark.parametrize("input_size", [2, 4])
def test_f(batch_size: int, neuron_number: int, input_size: int) -> None:
    m = CellModel(neuron_number, input_size)

    x_t = th.randn(batch_size, neuron_number)
    input_t = th.randn(batch_size, input_size)

    out = m(x_t, input_t)

    assert len(out.size()) == 2
    assert out.size(0) == batch_size
    assert out.size(1) == neuron_number


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("neuron_number", [2, 4])
@pytest.mark.parametrize("input_size", [2, 4])
@pytest.mark.parametrize("unfolding_steps", [2, 4])
def test_cell(
    batch_size: int, neuron_number: int, input_size: int, unfolding_steps: int
) -> None:
    c = LiquidCell(neuron_number, input_size, unfolding_steps)

    x_t = th.randn(batch_size, neuron_number)
    input_t = th.randn(batch_size, input_size)
    delta_t = th.rand(batch_size)

    out = c(x_t, input_t, delta_t)

    assert len(out.size()) == 2
    assert out.size(0) == batch_size
    assert out.size(1) == neuron_number


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("neuron_number", [2, 4])
@pytest.mark.parametrize("input_size", [2, 4])
@pytest.mark.parametrize("unfolding_steps", [2, 4])
@pytest.mark.parametrize("time_steps", [2, 4])
@pytest.mark.parametrize("output_size", [2, 4])
@pytest.mark.parametrize(
    "ltc_constructor", [LiquidRecurrentReg, LiquidRecurrent]
)
def test_recurrent(
    batch_size: int,
    neuron_number: int,
    input_size: int,
    unfolding_steps: int,
    time_steps: int,
    output_size: int,
    ltc_constructor: Type[LiquidRecurrent],
) -> None:
    r = ltc_constructor(
        neuron_number, input_size, unfolding_steps, output_size
    )

    input_t = th.randn(batch_size, input_size, time_steps)
    delta_t = th.rand(batch_size, time_steps)

    out = r(input_t, delta_t)

    assert len(out.size()) == 3
    assert out.size(0) == batch_size
    assert out.size(1) == output_size
    assert out.size(2) == time_steps


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("neuron_number", [2, 4])
@pytest.mark.parametrize("input_size", [2, 4])
@pytest.mark.parametrize("unfolding_steps", [2, 4])
@pytest.mark.parametrize("time_steps", [512, 256])
@pytest.mark.parametrize("output_size", [2, 4])
@pytest.mark.parametrize(
    "ltc_constructor", [LiquidRecurrentBrainActivity, LiquidRecurrentLast]
)
def test_recurrent_single(
    batch_size: int,
    neuron_number: int,
    input_size: int,
    unfolding_steps: int,
    time_steps: int,
    output_size: int,
    ltc_constructor: Type[LiquidRecurrent],
) -> None:
    r = ltc_constructor(
        neuron_number, input_size, unfolding_steps, output_size
    )

    input_t = th.randn(batch_size, input_size, time_steps)
    delta_t = th.rand(batch_size, time_steps)

    out = r(input_t, delta_t)

    assert len(out.size()) == 2
    assert out.size(0) == batch_size
    assert out.size(1) == output_size
