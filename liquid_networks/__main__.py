import argparse
import re

from .data import DatasetNames
from .eval import eval_main
from .networks import ActivationFunction, TaskType
from .options import EvalOptions, ModelOptions, TrainOptions
from .train import train_main


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

    # main parser
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
        default=[],
    )
    parser.add_argument("--cuda", action="store_true")

    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    # train parser
    train_parser = sub_parsers.add_parser("train")

    train_parser.add_argument("run_name", type=str)
    train_parser.add_argument("output_folder", type=str)
    train_parser.add_argument("--epoch", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--metric-window-size", type=int, default=64)
    train_parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DatasetNames)
    )
    train_parser.add_argument("--train-dataset-path", type=str, required=True)
    train_parser.add_argument("--valid-dataset-path", type=str)
    train_parser.add_argument("--save-every", type=int, default=1024)
    train_parser.add_argument("--eval-every", type=int, default=1024)

    # eval parser
    eval_parser = sub_parsers.add_parser("eval")

    eval_parser.add_argument("run_name", type=str)
    eval_parser.add_argument("output_folder", type=str)
    eval_parser.add_argument("--model-path", type=str, required=True)
    eval_parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DatasetNames)
    )
    eval_parser.add_argument("--dataset-path", type=str, required=True)
    eval_parser.add_argument("--batch-size", type=int, default=256)

    # get args
    args = parser.parse_args()

    model_options = ModelOptions(
        neuron_number=args.neuron_number,
        unfolding_steps=args.unfolding_steps,
        activation_function=args.activation_function,
        task_type=args.task_type,
        specific_parameters=dict(args.specific_parameters),
        cuda=args.cuda,
    )

    if args.mode == "train":
        train_options = TrainOptions(
            epoch=args.epoch,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_folder=args.output_folder,
            run_name=args.run_name,
            metric_window_size=args.metric_window_size,
            dataset_name=args.dataset,
            train_dataset_path=args.train_dataset_path,
            valid_dataset_path=args.valid_dataset_path,
            save_every=args.save_every,
            eval_every=args.eval_every,
        )

        train_main(model_options, train_options)

    elif args.mode == "eval":
        eval_options = EvalOptions(
            model_path=args.model_path,
            output_folder=args.output_folder,
            run_name=args.run_name,
            dataset_name=args.dataset,
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
        )

        eval_main(model_options, eval_options)

    else:
        parser.error(f"Unrecognized mode: {args.mode}")


if __name__ == "__main__":
    main()
