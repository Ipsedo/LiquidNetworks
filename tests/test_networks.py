import pytest
import torch as th

from liquid_networks.networks.liquid_cell import CellModel, LiquidCell
from liquid_networks.networks.recurrents.bfrb import BfrbLiquidRecurrent
from liquid_networks.networks.recurrents.brain_activity import BrainActivityLiquidRecurrent
from liquid_networks.networks.recurrents.simple import LiquidRecurrent
from liquid_networks.networks.recurrents.variants import LastLiquidRecurrent


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("neuron_number", [2, 4])
@pytest.mark.parametrize("input_size", [2, 4])
def test_f(batch_size: int, neuron_number: int, input_size: int) -> None:
    m = CellModel(neuron_number, input_size, th.tanh)

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
def test_cell(batch_size: int, neuron_number: int, input_size: int, unfolding_steps: int) -> None:
    c = LiquidCell(neuron_number, input_size, unfolding_steps, th.tanh, 1.0)

    x_t = th.randn(batch_size, neuron_number)
    input_t = th.randn(batch_size, input_size)

    out = c(x_t, input_t)

    assert len(out.size()) == 2
    assert out.size(0) == batch_size
    assert out.size(1) == neuron_number


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("neuron_number", [2, 4])
@pytest.mark.parametrize("input_size", [2, 4])
@pytest.mark.parametrize("unfolding_steps", [2, 4])
@pytest.mark.parametrize("time_steps", [2, 4])
@pytest.mark.parametrize("output_size", [2, 4])
def test_recurrent(
    batch_size: int,
    neuron_number: int,
    input_size: int,
    unfolding_steps: int,
    time_steps: int,
    output_size: int,
) -> None:
    r = LiquidRecurrent(neuron_number, input_size, unfolding_steps, th.tanh, 1.0, output_size)

    input_t = th.randn(batch_size, time_steps, input_size)

    out = r(input_t)

    assert len(out.size()) == 3
    assert out.size(0) == batch_size
    assert out.size(1) == time_steps
    assert out.size(2) == output_size


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("neuron_number", [2, 4])
@pytest.mark.parametrize("input_size", [2, 4])
@pytest.mark.parametrize("unfolding_steps", [2, 4])
@pytest.mark.parametrize("time_steps", [512, 256])
@pytest.mark.parametrize("output_size", [2, 4])
def test_recurrent_single(
    batch_size: int,
    neuron_number: int,
    input_size: int,
    unfolding_steps: int,
    time_steps: int,
    output_size: int,
) -> None:
    r = LastLiquidRecurrent(neuron_number, input_size, unfolding_steps, th.tanh, 1.0, output_size)

    input_t = th.randn(batch_size, time_steps, input_size)

    out = r(input_t)

    assert len(out.size()) == 2
    assert out.size(0) == batch_size
    assert out.size(1) == output_size


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("neuron_number", [2, 4])
@pytest.mark.parametrize("input_size", [2, 4])
@pytest.mark.parametrize("unfolding_steps", [2, 4])
@pytest.mark.parametrize("time_steps", [512, 256])
@pytest.mark.parametrize("nb_layer", [2, 4])
@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_recurrent_brain_activity(
    batch_size: int,
    neuron_number: int,
    input_size: int,
    unfolding_steps: int,
    time_steps: int,
    nb_layer: int,
    factor: float,
) -> None:
    r = BrainActivityLiquidRecurrent(
        neuron_number,
        input_size,
        unfolding_steps,
        th.tanh,
        1.0,
        nb_layer,
        factor,
    )

    input_t = th.randn(batch_size, time_steps, input_size)

    out = r(input_t)

    assert len(out.size()) == 2
    assert out.size(0) == batch_size
    assert out.size(1) == 6


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("neuron_number", [2, 4])
@pytest.mark.parametrize("unfolding_steps", [2, 4])
@pytest.mark.parametrize("time_steps", [512, 256])
def test_recurrent_bfrb(
    batch_size: int,
    neuron_number: int,
    unfolding_steps: int,
    time_steps: int,
) -> None:
    r = BfrbLiquidRecurrent(neuron_number, unfolding_steps, th.tanh, 1.0, 0.1)

    input_grids = th.randn(batch_size, time_steps, r.nb_grids, *r.grid_size)
    input_features = th.randn(batch_size, time_steps, r.nb_features)

    out = r((input_grids, input_features))

    assert len(out.size()) == 2
    assert out.size(0) == batch_size
    assert out.size(1) == r.output_size
