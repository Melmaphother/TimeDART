import torch
from argparse import Namespace
from args import get_args
from data_provider.data_factory import data_provider
from models.TimeDAR import (
    TimeDAR,
    TimeDARForecasting,
)
from exp.pretrain import Pretrain
from exp.finetune import ForecastingFinetune


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pretrain(args: Namespace):
    train_loader = data_provider(args, "train")
    val_loader = data_provider(args, "val")
    test_loader = data_provider(args, "test")

    model = TimeDAR(args)
    if args.verbose:
        print(f"Numbers of pretrain model parameters: {count_parameters(model)}")
    _pretrain = Pretrain(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    _pretrain.pretrain()
    _pretrain.pretrain_test()


def finetune(args: Namespace):
    train_loader = data_provider(args, "train")
    val_loader = data_provider(args, "val")
    test_loader = data_provider(args, "test")

    _pretrain_model_save_path = args.pretrain_save_dir / "pretrain_model.pth"
    pretrain_model = TimeDAR(args)
    if _pretrain_model_save_path.exists():
        if args.verbose:
            print(f"Load pretrain model from {_pretrain_model_save_path}")
        pretrain_model.load_state_dict(
            torch.load(_pretrain_model_save_path, map_location=args.device)
        )
        if args.verbose:
            print("Successfully loaded!")
    else:
        raise FileNotFoundError(
            f"Pretrain Model not found in {_pretrain_model_save_path}"
        )

    finetune_model = TimeDARForecasting(args=args, TimeDAR_encoder=pretrain_model)
    if args.verbose:
        print(
            f"Numbers of finetune model parameters: {count_parameters(finetune_model)}"
        )

    _finetune = ForecastingFinetune(
        args=args,
        model=finetune_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        is_fine_zero=False,
    )
    _finetune.finetune()
    mse, mae = _finetune.finetune_test()
    return mse, mae


def fine_zero(args: Namespace):
    train_loader = data_provider(args, "train")
    val_loader = data_provider(args, "val")
    test_loader = data_provider(args, "test")

    random_model = TimeDAR(args)
    fine_zero_model = TimeDARForecasting(args=args, TimeDAR_encoder=random_model)

    _fine_zero = ForecastingFinetune(
        args=args,
        model=fine_zero_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        is_fine_zero=True,
    )
    _fine_zero.finetune()
    mse, mae = _fine_zero.finetune_test()
    return mse, mae


def record_score(args: Namespace, ft_mse, ft_mae, fz_mse, fz_mae):
    """
    {input_len}->{pred_len}, finetune: {mse}, {mae}, fine_zero: {mse}, {mae}
    """
    record = f"{args.input_len}->{args.pred_len}, finetune: {ft_mse:.3f}, {ft_mae:.3f}, fine_zero: {fz_mse:.3f}, {fz_mae:.3f}\n"

    with open(args.scores_save_path, "a") as f:
        f.write(record)


if __name__ == "__main__":
    global_args = get_args()
    if global_args.verbose:
        system_info = (
            ">" * 15
            + f"{global_args.task_name}_{global_args.dataset}_{global_args.pretrain_settings}"
            + "<" * 15
            + "\n"
        )
        print(system_info)
        print(f"Arguments in Experiment: \n{global_args}\n")

    torch.manual_seed(global_args.seed)  # for reproducibility
    torch.cuda.empty_cache()

    if global_args.task_name == "pretrain":
        pretrain(global_args)
    elif global_args.task_name == "finetune":
        # ft_mse, ft_mae = finetune(global_args)
        fz_mse, fz_mae = fine_zero(global_args)
        # record_score(global_args, ft_mse, ft_mae, fz_mse, fz_mae)
        # fz_mse, fz_mae = fine_zero(global_args)
        # record_score(global_args, 0.0, 0.0, fz_mse, fz_mae)
    else:
        raise ValueError("Task name not found")
