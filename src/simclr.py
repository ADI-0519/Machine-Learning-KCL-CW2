import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        batch_size = z1.size(0)
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        positives = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size),
        ])

        denominator = torch.logsumexp(sim, dim=1)
        loss = -positives + denominator
        return loss.mean()


def train_simclr_epoch(model, loader, optimizer, criterion, device):
    """Run one SimCLR training epoch and return average contrastive loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for (x1, x2), itm in tqdm(loader, leave=False):
        x1, x2 = x1.to(device), x2.to(device)

        x, z1 = model(x1)
        y, z2 = model(x2)

        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x1.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)