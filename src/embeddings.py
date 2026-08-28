import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def grab_embeddings(encoder: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Extract, L2-normalize, and stack encoder embeddings."""
    encoder.eval()
    chunks: list[np.ndarray] = []

    for inputs, _targets in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        feats = encoder(inputs)
        feats = F.normalize(feats, p=2, dim=1)
        chunks.append(feats.cpu().numpy())

    if not chunks:
        raise ValueError("embedding loader produced no batches")
    return np.concatenate(chunks, axis=0)
