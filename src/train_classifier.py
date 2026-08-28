from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .models import CIFARClassifier
from .protocol import PROTOCOL_VERSION, EvaluationMetrics, TrainingOutcome


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Train a classifier for one epoch and return loss/accuracy metrics."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in tqdm(loader, leave=False, desc="Train"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimiser.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_examples += targets.size(0)

    if total_examples == 0:
        raise ValueError("training loader produced no examples")
    mean_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return {"loss": mean_loss, "accuracy": accuracy}


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate classifier on validation/test loader and return metrics."""
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

    if total_examples == 0:
        raise ValueError("evaluation loader produced no examples")
    mean_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return {"loss": mean_loss, "accuracy": accuracy}


def save_checkpoint(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    metrics: dict[str, float | int | str],
    checkpoint_path: str | Path,
) -> None:
    """Save model state and training metadata to a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, checkpoint_path)


def train_classifier(
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    verbose: bool = True,
) -> TrainingOutcome:
    """Train for fixed epochs, then evaluate the test loader exactly once."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

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

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimiser=optimiser,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "lr": optimiser.param_groups[0]["lr"],
        }
        history.append(epoch_summary)

        if verbose:
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"lr={optimiser.param_groups[0]['lr']:.6f}"
            )

    test_metrics = evaluate_classifier(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    metrics = EvaluationMetrics(
        trained_epochs=epochs,
        test_loss=test_metrics["loss"],
        test_accuracy=test_metrics["accuracy"],
    )

    if verbose:
        print(f"[Test] loss={metrics.test_loss:.4f} accuracy={metrics.test_accuracy:.4f}")

    if checkpoint_path is not None:
        save_checkpoint(
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            epoch=epochs,
            metrics={"protocol_version": PROTOCOL_VERSION, **metrics.to_dict()},
            checkpoint_path=checkpoint_path,
        )

    return TrainingOutcome(model=model, metrics=metrics, history=history)
