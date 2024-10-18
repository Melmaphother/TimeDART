import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import Namespace
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
from models.TimeDART import TimeDART, TimeDARTForecasting
from utils.tools import EarlyStopping, EpochTimer, adjust_learning_rate
import itertools


@dataclass
class Metrics:
    mse_loss: float = 0.0
    mae_loss: float = 0.0

    def __repr__(self):
        _repr = f"MSE: {self.mse_loss:.6f} | " f"MAE: {self.mae_loss:.6f}"
        return _repr


class ForecastingFinetune:
    def __init__(
        self,
        args: Namespace,
        model: TimeDARTForecasting,
        train_loader: DataLoader | list[DataLoader],
        val_loader: DataLoader | list[DataLoader],
        test_loader: DataLoader | list[DataLoader],
        is_fine_zero: bool,
    ):
        self.args = args
        self.verbose = args.verbose
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.finetune_mode = args.finetune_mode
        self.device = args.device

        # finetune mode
        if args.finetune_mode == "fine_all":
            self.model.unfreeze_encoder()
        elif args.finetune_mode == "fine_last":
            self.model.freeze_encoder()

        # Training Metrics
        self.num_epochs_finetune = args.num_epochs_finetune
        self.eval_per_epochs_finetune = args.eval_per_epochs_finetune
        self.mse_criterion = nn.MSELoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.finetune_lr,
        )
        if args.lr_adjust_method == "exp":
            self.scheduler = ExponentialLR(
                optimizer=self.optimizer,
                gamma=args.finetune_lr_decay,
            )
        else:
            if isinstance(self.train_loader, list):
                train_loader_len = sum([len(loader) for loader in self.train_loader])
            else:
                train_loader_len = len(self.train_loader)
            self.scheduler = OneCycleLR(
                optimizer=self.optimizer,
                steps_per_epoch=train_loader_len,
                pct_start=args.finetune_pct_start,
                epochs=self.num_epochs_finetune,
                max_lr=args.finetune_lr,
            )

        # Evaluation Metrics
        self.mae_criterion = nn.L1Loss()

        # Save Path
        if is_fine_zero:
            save_dir = args.finezero_save_dir
            self.train_result_save_path = save_dir / "finezero_train.txt"
            self.val_result_save_path = save_dir / "finezero_val.txt"
            self.test_result_save_path = save_dir / "finezero_test.txt"
            self.model_save_path = save_dir / "finezero_model.pth"
        else:
            save_dir = args.finetune_save_dir
            self.train_result_save_path = save_dir / "finetune_train.txt"
            self.val_result_save_path = save_dir / "finetune_val.txt"
            self.test_result_save_path = save_dir / "finetune_test.txt"
            self.model_save_path = save_dir / "finetune_model.pth"

    def finetune(self):
        self.train_result_save_path.touch()
        self.train_result_save_path.write_text("")
        self.val_result_save_path.touch()
        self.val_result_save_path.write_text("")
        early_stopping = EarlyStopping(self.args)
        epoch_timer = EpochTimer()

        for epoch in range(self.num_epochs_finetune):
            print(
                "Current learning rate: {:.7f}".format(
                    self.optimizer.param_groups[0]["lr"]
                )
            )

            epoch_timer.start()
            train_metrics = self.__train_one_epoch(epoch)
            with self.train_result_save_path.open("a") as f:
                f.write(f"Epoch: {epoch + 1} | {train_metrics}\n")
            if self.verbose:
                print(f"Finetune Train Epoch: {epoch + 1} | {train_metrics}")
            if self.args.lr_adjust_method != "step":
                adjust_learning_rate(
                    self.optimizer, self.scheduler, epoch + 1, self.args, verbose=True
                )
            else:
                print(f"Learning rate adjusted to {self.scheduler.get_last_lr()[0]}")

            epoch_timer.stop()
            if self.verbose:
                epoch_timer.print_duration(
                    epoch=epoch + 1, total_epochs=self.num_epochs_finetune
                )

            if (epoch + 1) % self.eval_per_epochs_finetune == 0:
                val_metrics = self.__val_one_epoch(self.val_loader)
                test_metrics = self.__val_one_epoch(self.test_loader)
                with self.val_result_save_path.open("a") as f:
                    f.write(
                        f"Epoch: {epoch + 1} | Val: {val_metrics} | Test: {test_metrics}\n"
                    )
                if self.verbose:
                    print(
                        f"Finetune Validate Epoch: {epoch + 1} | {val_metrics}", end=" "
                    )
                    print(f"Test | {test_metrics}")

                early_stopping(val_metrics.mse_loss, self.model, self.model_save_path)
                if early_stopping.early_stop:
                    if self.verbose:
                        print("Early stopping")
                    break

    def __train_one_epoch(self, epoch):
        self.model.train()
        metrics = Metrics()
        if isinstance(self.train_loader, list):
            train_loader = itertools.chain(*self.train_loader)
            train_loader_len = sum([len(loader) for loader in self.train_loader])
        else:
            train_loader = (
                tqdm(self.train_loader, desc="Finetune Train")
                if self.args.use_tqdm
                else self.train_loader
            )
            train_loader_len = len(self.train_loader)
        for x, y in train_loader:
            self.optimizer.zero_grad()
            y_pred = self.model(x, self.finetune_mode)
            mse_loss = self.mse_criterion(y_pred, y)
            mae_loss = self.mae_criterion(y_pred, y)
            mse_loss.backward()
            self.optimizer.step()
            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()

            if self.args.lr_adjust_method == "step":
                adjust_learning_rate(
                    self.optimizer, self.scheduler, epoch + 1, self.args, verbose=False
                )
                self.scheduler.step()

        metrics.mse_loss /= train_loader_len
        metrics.mae_loss /= train_loader_len
        return metrics

    @torch.no_grad()
    def __val_one_epoch(self, _val_loader: DataLoader):
        self.model.eval()
        metrics = Metrics()
        if isinstance(_val_loader, list):
            val_loader = itertools.chain(*_val_loader)
            val_loader_len = sum([len(loader) for loader in _val_loader])
        else:
            val_loader = (
                tqdm(_val_loader, desc="Finetune Validate")
                if self.args.use_tqdm
                else _val_loader
            )
            val_loader_len = len(_val_loader)
        for x, y in val_loader:
            y_pred = self.model(x, self.finetune_mode)
            mse_loss = self.mse_criterion(y_pred, y)
            mae_loss = self.mae_criterion(y_pred, y)
            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()

        metrics.mse_loss /= val_loader_len
        metrics.mae_loss /= val_loader_len
        return metrics

    @torch.no_grad()
    def finetune_test(self):
        self.test_result_save_path.touch()
        self.test_result_save_path.write_text("")

        model = TimeDARTForecasting(
            args=self.args, TimeDAR_encoder=TimeDART(args=self.args)
        ).to(self.device)
        if self.model_save_path.exists():
            if self.verbose:
                print(f"Load pretrain model from {self.model_save_path}")
            model.load_state_dict(torch.load(self.model_save_path))
            if self.verbose:
                print("Successfully loaded!")
        else:
            raise FileNotFoundError(f"Model not found in {self.model_save_path}")

        model.eval()
        metrics = Metrics()
        if isinstance(self.test_loader, list):
            test_loader = itertools.chain(*self.test_loader)
            test_loader_len = sum([len(loader) for loader in self.test_loader])
        else:
            test_loader = (
                tqdm(self.test_loader, desc="Finetune Test")
                if self.args.use_tqdm
                else self.test_loader
            )
            test_loader_len = len(self.test_loader)
        for x, y in test_loader:
            y_pred = model(x, self.finetune_mode)
            mse_loss = self.mse_criterion(y_pred, y)
            mae_loss = self.mae_criterion(y_pred, y)
            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()

        metrics.mse_loss /= test_loader_len
        metrics.mae_loss /= test_loader_len
        with self.test_result_save_path.open("w") as f:
            f.write(f"{metrics}\n")
        if self.verbose:
            print(f"Finetune Test | {metrics}")

        return metrics.mse_loss, metrics.mae_loss
