import argparse
import re
from typing import get_args

from .data import DatasetNames
from .networks import ActivationFunction, TaskType
from .options import ModelOptions, TrainOptions
from .train import train


def _parse_specific_parameters(arg: str) -> tuple[str, str]:
    regex_key_value = re.compile(r"^ *([^= ]+)=([^= ]+) *$")

    match = regex_key_value.match(arg)

    if match:
        key = match.group(1)
        value = match.group(2)
        return key, value
    raise argparse.ArgumentTypeError(
        f"invalid option: {arg}. Expected key-value pair, "
        "example: key_1=value_1"
    )


def main() -> None:
    parser = argparse.ArgumentParser("liquid_networks main")

    parser.add_argument("--neuron-number", type=int, default=32)
    parser.add_argument("--unfolding-steps", type=int, default=6)
    parser.add_argument(
        "--task-type", type=str, required=True, choices=list(TaskType)
    )
    parser.add_argument(
        "--activation-function",
        type=str,
        required=True,
        choices=list(ActivationFunction),
    )
    parser.add_argument(
        "-sp",
        "--specific-parameters",
        type=_parse_specific_parameters,
        action="append",
    )
    parser.add_argument("--cuda", action="store_true")

    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = sub_parsers.add_parser("train")

    train_parser.add_argument("run_name", type=str)
    train_parser.add_argument("output_folder", type=str)
    train_parser.add_argument("--epoch", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--metric-window-size", type=int, default=64)
    train_parser.add_argument(
        "--dataset",
        type=str,
        default="household_power",
        choices=get_args(DatasetNames),
    )
    train_parser.add_argument("--train-data-path", type=str, required=True)
    train_parser.add_argument("--valid-data-path", type=str)
    train_parser.add_argument("--save-every", type=int, default=1024)
    train_parser.add_argument("--eval-every", type=int, default=1024)

    args = parser.parse_args()

    model_options = ModelOptions(
        args.neuron_number,
        args.unfolding_steps,
        args.activation_function,
        args.task_type,
        dict(args.specific_parameters),
        args.cuda,
    )

    if args.mode == "train":
        train_options = TrainOptions(
            args.epoch,
            args.batch_size,
            args.learning_rate,
            args.output_folder,
            args.run_name,
            args.metric_window_size,
            args.dataset,
            args.train_data_path,
            args.valid_data_path,
            args.save_every,
            args.eval_every,
        )

        train(model_options, train_options)
    else:
        parser.error(f"Unrecognized mode: {args.mode}")


if __name__ == "__main__":
    main()
