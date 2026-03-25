from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import load_configurations
from src.data import SimCLRTransform, get_cifar10_train
from src.models import SimCLRModel
from src.seed import set_seed
from src.simclr import NTXentLoss, train_simclr_epoch


def main() -> None:
    cfg = load_configurations("configs/default.yaml")
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",device)
    simclr_cfg = cfg["simclr"]
    data_cfg = cfg["data"]

    dataset = get_cifar10_train(
        root=data_cfg["root"],
        transform=SimCLRTransform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=simclr_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    model = SimCLRModel(proj_dim=simclr_cfg["projection_dim"]).to(device)
    criterion = NTXentLoss(temperature=simclr_cfg["temperature"])
    optimizer = SGD(
        model.parameters(),
        lr=simclr_cfg["lr"],
        momentum=simclr_cfg["momentum"],
        nesterov=simclr_cfg.get("nesterov", False),
        weight_decay=simclr_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=simclr_cfg["epochs"],
        eta_min=simclr_cfg.get("min_lr", 1e-6),
    )

    save_path = Path(simclr_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, simclr_cfg["epochs"] + 1):
        loss = train_simclr_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[SimCLR Epoch {epoch:03d}/{simclr_cfg['epochs']:03d}] "
            f"loss={loss:.4f} lr={current_lr:.6f}"
        )

        if loss < best_loss:
            best_loss = loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                },
                save_path,
            )

    print(f"Saved best SimCLR checkpoint to {save_path}")


if __name__ == "__main__":
    main()
