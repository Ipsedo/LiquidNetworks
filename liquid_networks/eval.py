from os import makedirs
from os.path import exists, isdir
from typing import Callable

import mlflow
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import AbstractDataset
from .networks import AbstractLiquidRecurrent
from .networks.functions import LossFunctionType
from .options import EvalOptions, ModelOptions


def eval_model_on_dataset[T](
    ltc: AbstractLiquidRecurrent[T],
    valid_dataset: AbstractDataset[T],
    batch_size: int,
    device: th.device,
    loss_fn: LossFunctionType,
    dataloader_workers: int,
    callback_batch_iter: Callable[[int, int], None] | None = None,
) -> float:
    with th.no_grad():
        ltc.eval()

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=dataloader_workers,
            collate_fn=valid_dataset.collate_fn,
        )

        valid_loss = 0.0
        nb_valid_examples = 0

        for i, (f_v, y_v) in enumerate(valid_dataloader):
            f_v = valid_dataset.to_device(f_v, device)
            y_v = y_v.to(device=device)

            valid_loss += loss_fn(ltc(f_v, valid_dataset.delta_t), y_v, "sum").item()

            if callback_batch_iter is not None:
                callback_batch_iter(i, len(valid_dataset))

            nb_valid_examples += y_v.size(0)

        ltc.train()

        return valid_loss / nb_valid_examples


def eval_main(model_options: ModelOptions, eval_options: EvalOptions) -> None:
    if not exists(eval_options.output_folder):
        makedirs(eval_options.output_folder)
    elif not isdir(eval_options.output_folder):
        raise NotADirectoryError(eval_options.output_folder)

    device = model_options.get_device()

    with mlflow.start_run(run_name=eval_options.run_name):

        mlflow.log_params({**dict(model_options), **dict(eval_options)})

        print(f"Will load '{eval_options.dataset_name}' dataset.")

        dataset = eval_options.get_dataset()

        print("train data count:", len(dataset))

        assert dataset.task_type == model_options.task_type

        ltc = model_options.get_model()
        loss_fn = model_options.get_loss_function()

        ltc.to(device=device)
        ltc.load_state_dict(th.load(eval_options.model_path, map_location=device))

        print("Nb parameters:", ltc.count_parameters())

        tqdm_bar = tqdm(total=len(dataset))
        tqdm_bar.set_description("Evaluate")

        def _callback(_: int, __: int) -> None:
            tqdm_bar.update(eval_options.batch_size)

        eval_loss = eval_model_on_dataset(
            ltc, dataset, eval_options.batch_size, device, loss_fn, eval_options.workers, _callback
        )

        tqdm_bar.write(f"Eval loss = {eval_loss}")
        mlflow.log_metric("eval_loss", eval_loss)
