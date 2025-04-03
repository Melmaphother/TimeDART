from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    transfer_weights,
    show_series,
    show_matrix,
)
from utils.augmentations import masked_data
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


class Exp_TimeDART_v2(Exp_Basic):
    def __init__(self, args):
        super(Exp_TimeDART_v2, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

    def _build_model(self):
        if self.args.downstream_task == "forecast":
            model = self.model_dict[self.args.model].Model(self.args).float()
        elif self.args.downstream_task == "classification":
            model = self.model_dict[self.args.model].ClsModel(self.args).float()

        # if self.args.load_checkpoints:
        #     print(
        #         "Loading pre-trained checkpoint from: {}".format(
        #             self.args.load_checkpoints
        #         )
        #     )
        #     checkpoint = torch.load(
        #         self.args.load_checkpoints, map_location=self.device
        #     )
        #     model.load_state_dict(checkpoint["model_state_dict"])
        #     print(
        #         "Successfully loaded best pre-trained model from epoch {}".format(
        #             checkpoint["epoch"]
        #         )
        #     )

        # print out the model size
        print(
            "number of model params",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if (
            self.args.task_name == "finetune"
            and self.args.downstream_task == "classification"
        ):
            criterion = nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss")
        else:
            criterion = nn.MSELoss()
            print("Using MSELoss")
        return criterion

    def pretrain(self):
        # data preparation
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=model_optim, gamma=self.args.lr_decay
        )

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            # current learning rate
            print(
                "Current learning rate: {:.7f}".format(model_scheduler.get_last_lr()[0])
            )

            # Training phase with gradient accumulation
            train_loss = []
            model_criterion = self._select_criterion()
            self.model.train()

            # Initialize gradient accumulation
            accumulation_steps = self.args.accumulation_steps
            model_optim.zero_grad()

            train_loader = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{self.args.train_epochs}"
            )
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)
                diff_loss = model_criterion(pred_x, batch_x)
                # Scale loss by accumulation steps
                diff_loss = diff_loss / accumulation_steps
                diff_loss.backward()

                # Gradient accumulation
                if (i + 1) % accumulation_steps == 0:
                    model_optim.step()
                    model_optim.zero_grad()

                train_loss.append(
                    diff_loss.item() * accumulation_steps
                )  # Scale back for logging

            # Handle remaining gradients
            if (i + 1) % accumulation_steps != 0:
                model_optim.step()
                model_optim.zero_grad()

            model_scheduler.step()
            train_loss = np.mean(train_loss)

            # Validation phase
            vali_loss = self.valid_one_epoch(vali_loader)

            # log and Loss
            end_time = time.time()
            print(
                "Epoch: {}/{}, Time: {:.2f}, Train Loss: {:.4f}, Vali Loss: {:.4f}".format(
                    epoch + 1,
                    self.args.train_epochs,
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                )
            )

            loss_scalar_dict = {
                "train_loss": train_loss,
                "vali_loss": vali_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model epoch{}...".format(
                        min_vali_loss, vali_loss, epoch
                    )
                )
                min_vali_loss = vali_loss

                # Save best model with all parameters
                model_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                }
                torch.save(model_ckpt, os.path.join(path, f"ckpt_best.pth"))

            # Save checkpoint for every epoch
            print("Saving model at epoch {}...".format(epoch + 1))
            model_ckpt = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
            }
            torch.save(model_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def valid_one_epoch(self, vali_loader):
        vali_loss = []
        model_criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)
                diff_loss = model_criterion(pred_x, batch_x)
                vali_loss.append(diff_loss.item())

        vali_loss = np.mean(vali_loss)

        return vali_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_criteria = self._select_criterion()
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loader = tqdm(train_loader, desc="Training")

            print(
                "Current learning rate: {:.7f}".format(
                    model_optim.param_groups[0]["lr"]
                )
            )

            self.model.train()
            start_time = time.time()

            # Initialize gradient accumulation
            accumulation_steps = self.args.accumulation_steps
            model_optim.zero_grad()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x, batch_y)  # 训练时传入batch_y

                f_dim = -1 if self.args.features == "MS" else 0

                loss = model_criteria(pred_x, batch_y)
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
                loss.backward()

                # Gradient accumulation
                if (i + 1) % accumulation_steps == 0:
                    model_optim.step()
                    model_optim.zero_grad()

                train_loss.append(
                    loss.item() * accumulation_steps
                )  # Scale back for logging

                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

            # Handle remaining gradients
            if (i + 1) % accumulation_steps != 0:
                model_optim.step()
                model_optim.zero_grad()

            train_loss = np.mean(train_loss)
            vali_loss = self.valid(vali_loader, model_criteria)
            # test_loss = self.valid(test_loader, model_criteria)

            end_time = time.time()
            print(
                "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                    epoch + 1,
                    len(train_loader),
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                )
            )
            log_path = path + "/" + "log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f}\n".format(
                        epoch + 1,
                        len(train_loader),
                        end_time - start_time,
                        train_loss,
                        vali_loss,
                    )
                )
            early_stopping(vali_loss, self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))

        self.lr = model_scheduler.get_last_lr()[0]

        return self.model

    def valid(self, vali_loader, model_criteria):
        vali_loss = []
        self.model.eval()
        vali_loader = tqdm(vali_loader, desc="Validation")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x, batch_y)  # 验证时传入batch_y

                f_dim = -1 if self.args.features == "MS" else 0

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()

                loss = model_criteria(pred_x, batch_y)
                vali_loss.append(loss.item())

        vali_loss = np.mean(vali_loss)
        self.model.train()

        return vali_loss

    def test(self):
        test_data, test_loader = self._get_data(flag="test")

        preds = []
        trues = []

        folder_path = "./outputs/test_results/{}".format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        test_loader = tqdm(test_loader, desc="Testing")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)  # 测试时不传入batch_y，使用自回归生成

                f_dim = -1 if self.args.features == "MS" else 0

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, _, _, _ = metric(preds, trues)
        print(
            "{0}->{1}, mse:{2:.3f}, mae:{3:.3f}".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        f = open(folder_path + "/score.txt", "a")
        f.write(
            "{0}->{1}, {2:.3f}, {3:.3f} \n".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        f.close()
