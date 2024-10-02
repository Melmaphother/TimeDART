import torch
import time
from argparse import Namespace

def adjust_learning_rate(optimizer, scheduler, epoch, args, verbose):
    if args.lr_adjust_method == "decay":
        lr_adjust = {epoch: args.finetune_lr * (args.finetune_lr_decay ** ((epoch - 1) // 1)) }
    elif args.lr_adjust_method == "step":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lr_adjust_method == "exp":
        lr_adjust = {}
        scheduler.step()
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if verbose:
            print(f"Learning rate adjusted to {lr}.")


class EarlyStopping:
    def __init__(self, args: Namespace):
        self.patience = args.patience
        self.verbose = args.verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model, model_save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_save_path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), model_save_path)
        self.val_loss_min = val_loss


class EpochTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError(
                "Timer has not been started. Use start() to start the timer."
            )
        self.end_time = time.time()

    def reset(self):
        self.start_time = time.time()
        self.end_time = None

    def get_duration(self):
        if self.start_time is None:
            raise ValueError(
                "Timer has not been started. Use start() to start the timer."
            )
        if self.end_time is None:
            # Timer is still running
            duration = time.time() - self.start_time
        else:
            # Timer has been stopped
            duration = self.end_time - self.start_time
        if duration < 0:
            raise ValueError("Timer has not been stopped.")
        return duration

    def print_duration(self, epoch, total_epochs):
        duration = self.get_duration()
        print(f"Epoch {epoch}/{total_epochs} took {duration:.2f} seconds.")
