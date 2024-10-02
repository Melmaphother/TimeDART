import argparse
import torch
import json
from pathlib import Path


def adjust_epochs(_args: argparse.Namespace):
    _args.eval_per_epochs_pretrain = (
        _args.eval_per_epochs_pretrain
        if 0 < _args.eval_per_epochs_pretrain < _args.num_epochs_pretrain
        else 1
    )
    _args.eval_per_epochs_finetune = (
        _args.eval_per_epochs_finetune
        if 0 < _args.eval_per_epochs_finetune < _args.num_epochs_finetune
        else 1
    )
    if _args.task_name == "pretrain":
        _args.patience = (
            _args.patience if 0 < _args.patience < _args.num_epochs_pretrain else 1
        )
    elif _args.task_name == "finetune":
        _args.patience = (
            _args.patience if 0 < _args.patience < _args.num_epochs_finetune else 1
        )


def check_patch_args(_args: argparse.Namespace):
    # ensure input_len -  patch_len can be divided by stride
    if (_args.input_len - _args.patch_len) % _args.stride != 0:
        raise ValueError(
            f"Invalid args: {_args.input_len=}, {_args.patch_len=}, {_args.stride=}",
            "(input_len - patch_len) should be divided by stride",
        )


def organize_file_structure(_args: argparse.Namespace):
    """
    ./results
    ├── ETTh1
    │   ├── `pretrain_settings` like: bs16_i336_...
    │   │   ├── pretrain
    │   │   │   ├── pretrain_model.pth
    │   │   │   ├── pretrain_train.txt
    │   │   │   ├── pretrain_val.txt
    │   │   ├── finetune
    │   │   │   ├── `finetune_settings 96` like: pl96_hd0.0_fl0.0001_f_modefine_all
    │   │   │   │   ├── finetune_model.pth
    │   │   │   │   ├── finetune_train.txt
    │   │   │   │   ├── finetune_val.txt
    │   │   │   │   ├── finetune_test.txt
    │   │   │   ├── `finetune_settings 192`
    │   │   │   │   ├── ...
    │   │   │   ├── `finetune_settings 336`
    │   │   │   │   ├── ...
    │   │   │   ├── `finetune_settings 720`
    │   │   │   │   ├── ...
    │   │   ├── scores.txt  # save all **newest** test scores (MSE and MAE) for records, including pred_len [96, 192, 336, 720]
    │   │   ├── args.json  # save pretrain args of this experiment (finetune args are saved by default value here)
    │   ├── other_pretrain_settings
    │   ├── logs  # save all **newest** logs for tensorboard
    ├── ETTh2 or other datasets
    """
    save_dir = Path(_args.save_dir)  # results/
    if _args.task_name == "pretrain":
        save_dir = save_dir / _args.dataset  # results/ETTh1 or results/ETTh1-ETTh2
    elif _args.task_name == "finetune":
        save_dir = save_dir / _args.pretrain_dataset  # results/ETTh1-ETTh2
    # important settings will be added to save directory to distinguish different experiments
    pretrain_settings = "bs{}_i{}_p{}_s{}_pe{}_d{}_nh{}_fd{}_nlc{}_nld{}_ts{}_sc{}_hd{}_pl{}_pwd{}_pld{}_nep{}_epp{}".format(
        _args.train_batch_size,
        _args.input_len,
        _args.patch_len,
        _args.stride,
        _args.position_encoding,
        _args.d_model,
        _args.num_heads,
        _args.feedforward_dim,
        _args.num_layers_casual,
        _args.num_layers_denoising,
        _args.time_steps,
        _args.scheduler,
        _args.head_dropout,
        _args.pretrain_lr,
        _args.pretrain_weight_decay,
        _args.pretrain_lr_decay,
        _args.num_epochs_pretrain,
        _args.eval_per_epochs_pretrain,
    )

    # Experiment
    exp_save_dir = save_dir / pretrain_settings
    exp_save_dir.mkdir(parents=True, exist_ok=True)
    # - args record
    args_save_path = exp_save_dir / "args.json"
    with open(args_save_path, "w") as f:
        json.dump(vars(_args), f, indent=4)
    # - scores record
    scores_save_path = exp_save_dir / "scores.txt"
    _args.scores_save_path = scores_save_path

    # Logs
    log_dir = save_dir / "logs"
    _args.log_dir = log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Pretrain
    _args.pretrain_settings = pretrain_settings
    pretrain_save_dir = exp_save_dir / "pretrain"
    _args.pretrain_save_dir = pretrain_save_dir

    # Finetune
    finetune_settings = "pl{}_nef{}_epf{}_fhd{}_fl{}_lam{}_fps{}_fld{}_f_mode{}".format(
        _args.pred_len,
        _args.num_epochs_finetune,
        _args.eval_per_epochs_finetune,
        _args.finetune_head_dropout,
        _args.finetune_lr,
        _args.lr_adjust_method,
        _args.finetune_pct_start,
        _args.finetune_lr_decay,
        _args.finetune_mode,
    )
    if _args.pretrain_dataset != _args.dataset:
        finetune_settings = f"{_args.dataset}_{finetune_settings}"
    finetune_save_dir = exp_save_dir / "finetune" / finetune_settings
    _args.finetune_save_dir = finetune_save_dir
    finezero_save_dir = exp_save_dir / "finezero" / finetune_settings
    _args.finezero_save_dir = finezero_save_dir

    # Create directories and files
    if _args.task_name == "pretrain":
        pretrain_save_dir.mkdir(parents=True, exist_ok=True)
    elif _args.task_name == "finetune":
        finetune_save_dir.mkdir(parents=True, exist_ok=True)
        scores_save_path.touch(exist_ok=True)
        finezero_save_dir.mkdir(parents=True, exist_ok=True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TimeDAR Args")

    parser.add_argument(
        "--model_name",
        type=str,
        default="DearCats",
        help="model name",
    )

    # Input/Output Directory
    parser.add_argument(
        "--data_dir", type=str, default="datasets", help="data directory"
    )
    parser.add_argument("--dataset", type=str, default="ETTh1", help="dataset name")
    parser.add_argument(
        "--pretrain_dataset", type=str, default="ETTh1", help="pretrain dataset name"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="save results or pretrain models directory",
    )

    # DataLoader
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="batch size for training"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=16, help="batch size for validation"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="batch size for testing"
    )
    parser.add_argument("--scale", action="store_true", help="scale data", default=True)

    # Data Features, (batch_size, seq_len, num_features)
    parser.add_argument("--input_len", type=int, default=336, help="sequence length")
    parser.add_argument(
        "--num_features", type=int, default=7, help="number of features or channels"
    )

    # Model Hyperparameters
    # - Transformer Encoder
    parser.add_argument("--position_encoding", type=str, default="absolute")
    parser.add_argument(
        "--embedding",
        type=str,
        default="patch",
        help="embedding method: token or patch",
    )
    parser.add_argument("--d_model", type=int, default=32, help="dimension of model")
    parser.add_argument("--num_heads", type=int, default=4, help="number of heads")
    parser.add_argument(
        "--feedforward_dim", type=int, default=64, help="dimension of feedforward"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument(
        "--num_layers_casual",
        type=int,
        default=2,
        help="number of layers in casual transformer",
    )

    # - Patch
    parser.add_argument("--patch_len", type=int, default=24, help="patch: patch length")
    parser.add_argument("--stride", type=int, default=24, help="patch: stride")

    # - Diffusion
    parser.add_argument(
        "--time_steps", type=int, default=1000, help="time steps in diffusion"
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", help="scheduler in diffusion"
    )

    # - Denoising Patch Decoder
    parser.add_argument(
        "--num_layers_denoising",
        type=int,
        default=1,
        help="number of layers in patch denoising decoder",
    )

    # Task and Training
    parser.add_argument(
        "--task_name",
        type=str,
        default="pretrain",
        help="task name: pretrain, finetune, finezero",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="early stopping patience",
    )
    # - Pretrain
    parser.add_argument(
        "--num_epochs_pretrain",
        type=int,
        default=30,
        help="number of epochs for pretrain",
    )
    parser.add_argument(
        "--eval_per_epochs_pretrain",
        type=int,
        default=1,
        help="evaluation per epochs for pretrain",
    )
    parser.add_argument(
        "--pretrain_lr", type=float, default=1e-3, help="pretrain learning rate"
    )
    parser.add_argument(
        "--pretrain_weight_decay",
        type=float,
        default=1e-3,
        help="pretrain weight decay",
    )
    parser.add_argument(
        "--pretrain_lr_decay",
        type=float,
        default=0.99,
        help="pretrain learning rate decay",
    )

    # - Finetune
    parser.add_argument(
        "--finetune_lr", type=float, default=1e-4, help="finetune learning rate"
    )
    parser.add_argument(
        "--finetune_weight_decay",
        type=float,
        default=1e-3,
        help="finetune weight decay",
    )
    parser.add_argument(
        "--finetune_pct_start",
        type=float,
        default=0.3,
        help="finetune pct start",
    )
    parser.add_argument(
        "--finetune_lr_decay",
        type=float,
        default=0.5,
        help="finetune learning rate decay",
    )
    parser.add_argument(
        "--lr_adjust_method",
        type=str,
        default="decay",
        help="learning rate decay method in finetune",
    )
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="fine_all",
        help="finetune mode: fine_all or fine_last",
    )
    parser.add_argument(
        "--num_epochs_finetune",
        type=int,
        default=10,
        help="number of epochs for finetune",
    )
    parser.add_argument(
        "--eval_per_epochs_finetune",
        type=int,
        default=1,
        help="evaluation per epochs for finetune",
    )
    # -- Forecasting Head
    parser.add_argument(
        "--downstream_task",
        type=str,
        default="forecasting",
        help="downstream task: forecasting or classification",
    )
    parser.add_argument("--pred_len", type=int, default=96, help="prediction length")
    parser.add_argument(
        "--head_dropout", type=float, default=0.1, help="pretrain head dropout"
    )
    parser.add_argument(
        "--finetune_head_dropout",
        type=float,
        default=0.0,
        help="forecasting head dropout",
    )

    # System
    parser.add_argument("--verbose", action="store_true", help="verbose", default=True)
    parser.add_argument(
        "--use_tqdm", action="store_true", help="use tqdm", default=False
    )
    parser.add_argument("--seed", type=int, default=2024, help="fixed random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device",
    )

    _args = parser.parse_args()
    adjust_epochs(_args)
    check_patch_args(_args)
    organize_file_structure(_args)

    # Device
    _args.device = torch.device(_args.device)
    return _args
