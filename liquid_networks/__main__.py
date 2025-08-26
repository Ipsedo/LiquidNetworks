import argparse
import re

from .data import DatasetNames
from .networks import ActivationFunction, TaskType
from .options import EvalOptions, ModelOptions, TrainOptions
from .predict import predict_main
from .train import train_main


def _parse_key_value_parameters(arg: str) -> tuple[str, str]:
    regex_key_value = re.compile(r"^ *([^= ]+)=([^= ]+) *$")

    match = regex_key_value.match(arg)

    if match:
        key = match.group(1)
        value = match.group(2)
        return key, value

    raise argparse.ArgumentTypeError(
        f"invalid option: {arg}. Expected key-value pair, " "example: key_1=value_1"
    )


def main() -> None:
    parser = argparse.ArgumentParser("liquid_networks main")

    # main parser
    parser.add_argument("--seed", type=int, default=314159)

    parser.add_argument("--neuron-number", type=int, default=32)
    parser.add_argument("--unfolding-steps", type=int, default=6)
    parser.add_argument("--delta-t", type=float, default=1.0)
    parser.add_argument("--task-type", type=str, required=True, choices=list(TaskType))
    parser.add_argument(
        "--activation-function", type=str, required=True, choices=list(ActivationFunction)
    )
    parser.add_argument(
        "-mp",
        "--model-parameters",
        type=_parse_key_value_parameters,
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
    train_parser.add_argument("--dataset", type=str, required=True, choices=list(DatasetNames))
    train_parser.add_argument(
        "-dp", "--dataset-parameters", type=_parse_key_value_parameters, action="append", default=[]
    )
    train_parser.add_argument("--train-dataset-path", type=str, required=True)
    train_parser.add_argument("--valid-dataset-path", type=str)
    train_parser.add_argument("--save-every", type=int, default=1024)
    train_parser.add_argument("--eval-every", type=int, default=1024)
    train_parser.add_argument("--dataloader-workers", type=int, default=4)

    # eval parser
    predict_parser = sub_parsers.add_parser("predict")

    predict_parser.add_argument("run_name", type=str)
    predict_parser.add_argument("output_folder", type=str)
    predict_parser.add_argument("--model-path", type=str, required=True)
    predict_parser.add_argument("--dataset", type=str, required=True, choices=list(DatasetNames))
    predict_parser.add_argument(
        "-dp", "--dataset-parameters", type=_parse_key_value_parameters, action="append", default=[]
    )
    predict_parser.add_argument("--dataset-path", type=str, required=True)
    predict_parser.add_argument("--batch-size", type=int, default=256)
    predict_parser.add_argument("--dataloader-workers", type=int, default=4)

    # get args
    args = parser.parse_args()

    model_options = ModelOptions(
        neuron_number=args.neuron_number,
        unfolding_steps=args.unfolding_steps,
        delta_t=args.delta_t,
        activation_function=args.activation_function,
        task_type=args.task_type,
        model_parameters=dict(args.model_parameters),
        cuda=args.cuda,
    )

    if args.mode == "train":
        train_options = TrainOptions(
            seed=args.seed,
            epoch=args.epoch,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_folder=args.output_folder,
            run_name=args.run_name,
            metric_window_size=args.metric_window_size,
            dataset_name=args.dataset,
            dataset_parameters=dict(args.dataset_parameters),
            train_dataset_path=args.train_dataset_path,
            valid_dataset_path=args.valid_dataset_path,
            save_every=args.save_every,
            eval_every=args.eval_every,
            workers=args.dataloader_workers,
        )

        train_main(model_options, train_options)

    elif args.mode == "predict":
        eval_options = EvalOptions(
            model_path=args.model_path,
            output_folder=args.output_folder,
            run_name=args.run_name,
            dataset_name=args.dataset,
            dataset_parameters=dict(args.dataset_parameters),
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            workers=args.dataloader_workers,
        )

        predict_main(model_options, eval_options)

    else:
        parser.error(f"Unrecognized mode: {args.mode}")


if __name__ == "__main__":
    main()
