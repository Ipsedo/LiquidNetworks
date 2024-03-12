# -*- coding: utf-8 -*-
import argparse
from typing import get_args

from .data import DatasetNames
from .networks import TaskType
from .options import ModelOptions, TrainOptions
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser("liquid_networks main")

    parser.add_argument("--neuron-number", type=int, default=32)
    parser.add_argument("--unfolding-steps", type=int, default=6)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--output-size", type=int, required=True)
    parser.add_argument(
        "--task-type", type=str, required=True, choices=get_args(TaskType)
    )
    parser.add_argument("--cuda", action="store_true")

    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = sub_parsers.add_parser("train")

    train_parser.add_argument("run_name", type=str)
    train_parser.add_argument("output_folder", type=str)
    train_parser.add_argument("--epoch", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--metric-window-size", type=int, default=64)
    train_parser.add_argument(
        "--dataset",
        type=str,
        default="household_power",
        choices=get_args(DatasetNames),
    )
    train_parser.add_argument("--data-path", type=str, required=True)

    args = parser.parse_args()

    model_options = ModelOptions(
        args.neuron_number,
        args.input_size,
        args.unfolding_steps,
        args.output_size,
        args.task_type,
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
            args.data_path,
        )
        train(model_options, train_options)
    else:
        parser.error(f"Unrecognized mode: {args.mode}")


if __name__ == "__main__":
    main()
