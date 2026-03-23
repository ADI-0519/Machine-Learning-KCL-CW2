from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .models import CIFARClassifier

def train_one_epoch(model:nn.Module, loader: torch.utils.data.DataLoader,optimiser: torch.optim.Optimizer,criterion: nn.Module,device: torch.device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images,targets in tqdm(loader, leave=False,desc="Train"):
        images = images.to(device,non_blocking=True)
        targets = targets.to(device,non_blocking=True)

        optimiser.zero_grad()
        logits = model(images)
        loss = criterion(logits,targets)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_examples += targets.size(0)

    mean_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)

    return {"loss": mean_loss,"accuracy": accuracy}

@torch.no_grad()
def evaluate_classifier(model: nn.Module,loader: torch.utils.data.DataLoader,criterion: nn.Module,device: torch.device) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in tqdm(loader, leave=False, desc="Eval"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_examples += targets.size(0)

    mean_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)

    return {"loss": mean_loss,"accuracy": accuracy}


def save_checkpoint(model: nn.Module,optimiser: torch.optim.Optimizer,scheduler: torch.optim.lr_scheduler.LRScheduler | None,epoch: int,metrics: dict[str, float],checkpoint_path: str | Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"epoch": epoch,"model_state_dict": model.state_dict(),"optimiser_state_dict": optimiser.state_dict(),"metrics": metrics}

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, checkpoint_path)


def train_classifier(train_loader: torch.utils.data.DataLoader,test_loader: torch.utils.data.DataLoader,num_classes: int,epochs: int,lr: float,momentum: float,weight_decay: float,device: torch.device,checkpoint_path: str | Path | None = None,verbose: bool = True) -> dict[str, Any]:
    model = CIFARClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs)

    history: list[dict[str, float | int]] = []
    best_test_accuracy = -1.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimiser=optimiser,
            criterion=criterion,
            device=device,
        )

        test_metrics = evaluate_classifier(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "lr": optimiser.param_groups[0]["lr"],
        }
        history.append(epoch_summary)

        if verbose:
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"test_loss={test_metrics['loss']:.4f} "
                f"test_acc={test_metrics['accuracy']:.4f}"
            )

        if test_metrics["accuracy"] > best_test_accuracy:
            best_test_accuracy = test_metrics["accuracy"]
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            if checkpoint_path is not None:
                save_checkpoint(
                    model=model,
                    optimiser=optimiser,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics={
                        "best_test_accuracy": best_test_accuracy,
                        "test_loss": test_metrics["loss"],
                    },
                    checkpoint_path=checkpoint_path,
                )

    if best_state_dict is None:
        raise RuntimeError("No best model state was recorded during training.")

    model.load_state_dict(best_state_dict)

    final_test_metrics = evaluate_classifier(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    return {
        "model": model,
        "best_test_accuracy": best_test_accuracy,
        "best_epoch": best_epoch,
        "final_test_loss": final_test_metrics["loss"],
        "final_test_accuracy": final_test_metrics["accuracy"],
        "history": history,
    }