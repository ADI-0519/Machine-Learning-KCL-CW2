import math

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("temperature must be a positive finite number")
        if not math.isfinite(float(temperature)) or temperature <= 0.0:
            raise ValueError("temperature must be a positive finite number")
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape or z1.shape[0] == 0:
            raise ValueError("z1 and z2 must be equally shaped non-empty feature matrices")
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        batch_size = z1.size(0)
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        positives = torch.cat(
            [
                torch.diag(sim, batch_size),
                torch.diag(sim, -batch_size),
            ]
        )

        denominator = torch.logsumexp(sim, dim=1)
        loss = -positives + denominator
        return loss.mean()


def train_simclr_epoch(model, loader, optimizer, criterion, device) -> float:
    """Run one SimCLR training epoch and return average contrastive loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for (x1, x2), _targets in tqdm(loader, leave=False):
        x1, x2 = x1.to(device), x2.to(device)

        _, z1 = model(x1)
        _, z2 = model(x2)

        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x1.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("SimCLR loader produced no training examples")
    return total_loss / total_examples
