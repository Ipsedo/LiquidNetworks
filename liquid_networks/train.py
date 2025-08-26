import random
from functools import partial
from os import makedirs
from os.path import exists, isdir

import mlflow
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm

from .eval import eval_model_on_dataset
from .metrics import Metric
from .options import ModelOptions, TrainOptions
from .saver import ModelSaver


def train_main(model_options: ModelOptions, train_options: TrainOptions) -> None:
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    th.manual_seed(train_options.seed)
    random.seed(train_options.seed)

    if not exists(train_options.output_folder):
        makedirs(train_options.output_folder)
    elif not isdir(train_options.output_folder):
        raise NotADirectoryError(train_options.output_folder)

    device = model_options.get_device()

    with mlflow.start_run(run_name=train_options.run_name):

        mlflow.log_params({**dict(model_options), **dict(train_options)})

        print(f"Will load '{train_options.dataset_name}' dataset.")

        train_dataset = train_options.get_train_dataset()
        valid_dataset = train_options.get_valid_dataset()

        print("train data count:", len(train_dataset))
        if valid_dataset is not None:
            print("eval data count:", len(valid_dataset))

        assert (
            train_dataset.task_type == model_options.task_type
        ), f"Wrong task type : '{train_dataset.task_type}' != '{model_options.task_type}'"

        ltc = model_options.get_model()
        optim = th.optim.Adam(ltc.parameters(), lr=train_options.learning_rate)
        loss_fn = model_options.get_loss_function()

        ltc.to(device=device)

        model_saver = ModelSaver(
            "ltc", train_options.output_folder, ltc, optim, train_options.save_every
        )

        print("Nb parameters:", ltc.count_parameters())

        loss_metric = Metric(train_options.metric_window_size)
        valid_metric = Metric(1)

        tqdm_bar = tqdm(range(train_options.epoch * len(train_dataset) // train_options.batch_size))

        def __callback(eval_batch_idx: int, nb_data: int, desc: str) -> None:
            tqdm_bar.set_description(
                f"{desc}Eval {eval_batch_idx} / {nb_data // train_options.batch_size}"
            )

        for e in range(train_options.epoch):

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_options.batch_size,
                shuffle=True,
                num_workers=train_options.workers,
                drop_last=True,
                collate_fn=train_dataset.collate_fn,
            )

            for f, y in train_dataloader:

                f = train_dataset.to_device(f, device)
                y = y.to(device=device)

                out = ltc(f)
                loss = loss_fn(out, y, "batchmean")

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                loss_metric.add_result(loss.item())
                grad_norm = ltc.grad_norm()

                mlflow.log_metrics(
                    {
                        "loss": loss_metric.get_last_metric(),
                        "loss_smoothed": loss_metric.get_smoothed_metric(),
                        "grad_norm": grad_norm,
                    },
                    step=tqdm_bar.n,
                )

                tqdm_bar.set_description(
                    f"Epoch {e:03} : "
                    f"loss = {loss_metric.get_last_metric():.4f}, "
                    f"loss_smoothed = {loss_metric.get_smoothed_metric():.4f}, "
                    f"valid_loss = {valid_metric.get_last_metric():.4f}, "
                    f"grad_norm = {grad_norm:.4f}"
                )

                model_saver.tick_save()

                if valid_dataset is not None and tqdm_bar.n % train_options.eval_every == 0:
                    valid_loss = eval_model_on_dataset(
                        ltc,
                        valid_dataset,
                        train_options.batch_size,
                        device,
                        loss_fn,
                        train_options.workers,
                        partial(__callback, desc=tqdm_bar.desc),
                    )

                    valid_metric.add_result(valid_loss)
                    mlflow.log_metric("valid_loss", valid_metric.get_last_metric(), step=tqdm_bar.n)

                tqdm_bar.update(1)
