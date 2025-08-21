from os import makedirs
from os.path import exists, isdir, join

import mlflow
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import Metric
from .options import ModelOptions, TrainOptions


def train(model_options: ModelOptions, train_options: TrainOptions) -> None:
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    if not exists(train_options.output_folder):
        makedirs(train_options.output_folder)
    elif not isdir(train_options.output_folder):
        raise NotADirectoryError(train_options.output_folder)

    if model_options.cuda:
        device = th.device("cuda")
        th.backends.cudnn.benchmark = True
    else:
        device = th.device("cpu")

    with mlflow.start_run(run_name=train_options.run_name):

        mlflow.log_params(
            {
                **model_options.to_dict(),
                **train_options.to_dict(),
            }
        )

        print(f"Will load '{train_options.dataset_name}' dataset.")

        train_dataset = train_options.get_train_dataset()
        valid_dataset = train_options.get_valid_dataset()

        assert train_dataset.task_type == model_options.task_type

        ltc = model_options.get_model()
        optim = th.optim.Adam(ltc.parameters(), lr=train_options.learning_rate)
        loss_fn = model_options.get_loss_function()

        ltc.to(device=device)

        loss_metric = Metric(train_options.metric_window_size)
        valid_metric = Metric(1)

        tqdm_bar = tqdm(
            range(
                train_options.epoch
                * len(train_dataset)
                // train_options.batch_size
            )
        )

        for e in range(train_options.epoch):

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_options.batch_size,
                shuffle=True,
                num_workers=12,
                drop_last=True,
                collate_fn=train_dataset.collate_fn,
            )

            for f, t, y in train_dataloader:

                f = train_dataset.to_device(f, device)
                t = t.to(device=device)
                y = y.to(device=device)

                out = ltc(f, t)
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

                tqdm_description = (
                    f"Epoch {e:03} : "
                    f"loss = {loss_metric.get_last_metric():.4f}, "
                    f"loss_smoothed = {loss_metric.get_smoothed_metric():.4f}, "
                    f"valid_loss = {valid_metric.get_last_metric():.4f}, "
                    f"grad_norm = {grad_norm:.4f}"
                )
                tqdm_bar.set_description(tqdm_description)

                if tqdm_bar.n % train_options.save_every == 0:
                    th.save(
                        ltc.state_dict(),
                        join(
                            train_options.output_folder, f"ltc_{tqdm_bar.n}.pt"
                        ),
                    )

                    th.save(
                        optim.state_dict(),
                        join(
                            train_options.output_folder,
                            f"optim_{tqdm_bar.n}.pt",
                        ),
                    )

                if (
                    valid_dataset is not None
                    and tqdm_bar.n % train_options.eval_every
                    == train_options.eval_every - 1
                ):
                    with th.no_grad():
                        ltc.eval()

                        valid_dataloader = DataLoader(
                            valid_dataset,
                            batch_size=train_options.batch_size,
                            num_workers=6,
                            collate_fn=valid_dataset.collate_fn,
                        )

                        valid_loss = 0.0
                        nb_valid_examples = 0

                        for i, (f_v, t_v, y_v) in enumerate(valid_dataloader):
                            f_v = valid_dataset.to_device(f_v, device)
                            t_v = t_v.to(device=device)
                            y_v = y_v.to(device=device)

                            valid_loss += loss_fn(
                                ltc(f_v, t_v), y_v, "sum"
                            ).item()

                            tqdm_bar.set_description(
                                f"{tqdm_description}, "
                                f"Eval {i} / {len(valid_dataset) // train_options.batch_size}"
                            )

                            nb_valid_examples += t_v.size(0)

                        valid_loss /= nb_valid_examples
                        valid_metric.add_result(valid_loss)
                        mlflow.log_metric(
                            "valid_loss",
                            valid_metric.get_last_metric(),
                            step=tqdm_bar.n,
                        )

                        ltc.train()

                tqdm_bar.update(1)
