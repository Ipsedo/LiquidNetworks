# -*- coding: utf-8 -*-
import argparse

from .options import ModelOptions, TrainOptions
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser("liquid_netorks main")

    parser.add_argument("--neuron-number", type=int, default=32)
    parser.add_argument("--unfolding-steps", type=int, default=16)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--output-size", type=int, required=True)

    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = sub_parsers.add_parser("train")

    train_parser.add_argument("run_name", type=str, required=True)
    train_parser.add_argument("output_folder", type=str, required=True)
    train_parser.add_argument("--epoch", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--metric-window-sizee", type=int, default=64)
    train_parser.add_argument("--cuda", type=bool, action="store_true")

    args = parser.parse_args()

    model_options = ModelOptions(
        args.neuron_number,
        args.input_size,
        args.unfolding_steps,
        args.output_size,
    )

    if args.mode == "train":
        train_options = TrainOptions(
            args.epoch,
            args.batch_size,
            args.learning_rate,
            args.output_folder,
            args.run_name,
            args.metric_window_size,
            args.cuda,
        )
        train(model_options, train_options)
    else:
        parser.error(f"Unrecognized mode: {args.mode}")


if __name__ == "__main__":
    main()
