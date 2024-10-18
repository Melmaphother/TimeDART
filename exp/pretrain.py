import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import Namespace
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models.TimeDART import TimeDART
from utils.tools import EarlyStopping, EpochTimer
import itertools


@dataclass
class Metrics:
    diff_loss: float = 0.0

    def __repr__(self):
        _repr = f"Diff Loss: {self.diff_loss:.6f}"
        return _repr


class Pretrain:
    def __init__(
        self,
        args: Namespace,
        model: TimeDART,
        train_loader: DataLoader | list[DataLoader],
        val_loader: DataLoader | list[DataLoader],
        test_loader: DataLoader | list[DataLoader],
    ):
        self.args = args
        self.verbose = args.verbose
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = args.device

        # Loss Function
        self.diff_criteria = nn.MSELoss()

        # Training Metrics
        self.num_epochs_pretrain = args.num_epochs_pretrain
        self.eval_per_epochs_pretrain = args.eval_per_epochs_pretrain
        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.pretrain_lr,
        )
        self.scheduler = ExponentialLR(
            optimizer=self.optimizer, gamma=args.pretrain_lr_decay
        )

        # Save result
        save_dir = args.pretrain_save_dir
        self.train_result_save_path = save_dir / "pretrain_train.txt"
        self.val_result_save_path = save_dir / "pretrain_val.txt"
        self.test_result_save_path = save_dir / "pretrain_test.txt"
        self.model_save_path = save_dir / "pretrain_model.pth"

    def pretrain(self):
        # create file and clear content
        self.train_result_save_path.touch(exist_ok=True)
        self.train_result_save_path.write_text("")
        self.val_result_save_path.touch(exist_ok=True)
        self.val_result_save_path.write_text("")
        early_stopping = EarlyStopping(self.args)
        epoch_timer = EpochTimer()

        for epoch in range(self.num_epochs_pretrain):
            epoch_timer.start()
            train_metrics = self.__train_one_epoch()
            # print training result to file
            with self.train_result_save_path.open("a") as f:
                f.write(f"Epoch: {epoch + 1} | {train_metrics}\n")
            if self.verbose:
                print(f"Train Epoch: {epoch + 1} | {train_metrics}")
                last_lr = self.scheduler.get_last_lr()[0]
                print(f"Updated Learning Rate to: {last_lr:.3e}")
            
            epoch_timer.stop()
            if self.verbose:
                epoch_timer.print_duration(epoch=epoch + 1, total_epochs=self.num_epochs_pretrain)

            if (epoch + 1) % self.eval_per_epochs_pretrain == 0:
                val_metrics = self.__val_one_epoch()
                # print validation result to file
                with self.val_result_save_path.open("a") as f:
                    f.write(f"Epoch: {epoch + 1} | {val_metrics}\n")
                if self.verbose:
                    print(f"Validate Epoch: {epoch + 1} | {val_metrics}")

                early_stopping(val_metrics.diff_loss, self.model, self.model_save_path)
                if early_stopping.early_stop:
                    if self.verbose:
                        print("Early stopping")
                    break

    def __train_one_epoch(self):
        self.model.train()
        metrics = Metrics()
        if isinstance(self.train_loader, list):
            train_loader = itertools.chain(*self.train_loader)
            train_loader_len = sum([len(loader) for loader in self.train_loader])
        else:
            train_loader = (
                tqdm(self.train_loader, desc="Training")
                if self.args.use_tqdm
                else self.train_loader
            )
            train_loader_len = len(self.train_loader)
        for x, _ in train_loader:
            self.optimizer.zero_grad()
            pred_x = self.model(x, task="pretrain")
            diff_loss = self.diff_criteria(pred_x, x)
            diff_loss.backward()
            self.optimizer.step()
            metrics.diff_loss += diff_loss.item()

        self.scheduler.step()
        metrics.diff_loss /= train_loader_len
        return metrics

    @torch.no_grad()
    def __val_one_epoch(self):
        self.model.eval()
        metrics = Metrics()
        if isinstance(self.val_loader, list):
            val_loader = itertools.chain(*self.val_loader)
            val_loader_len = sum([len(loader) for loader in self.val_loader])
        else:
            val_loader = (
                tqdm(self.val_loader, desc="Validation")
                if self.args.use_tqdm
                else self.val_loader
            )
            val_loader_len = len(self.val_loader)
        for x, _ in val_loader:
            pred_x = self.model(x, task="pretrain")
            diff_loss = self.diff_criteria(pred_x, x)
            metrics.diff_loss += diff_loss.item()

        metrics.diff_loss /= val_loader_len
        return metrics

    @torch.no_grad()
    def pretrain_test(self):
        self.test_result_save_path.touch(exist_ok=True)
        self.test_result_save_path.write_text("")

        model = TimeDART(self.args).to(self.device)
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
                tqdm(self.test_loader, desc="Testing")
                if self.args.use_tqdm
                else self.test_loader
            )
            test_loader_len = len(self.test_loader)
        for x, _ in test_loader:
            pred_x = model(x, task="pretrain")
            diff_loss = self.diff_criteria(pred_x, x)
            metrics.diff_loss += diff_loss.item()

        metrics.diff_loss /= test_loader_len
        # print test result to file
        with self.test_result_save_path.open("w") as f:
            f.write(f"{metrics}\n")
        if self.verbose:
            print(f"Test | {metrics}")
