import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.no_grad()
def grab_embeddings(encoder,loader:DataLoader,device:torch.device) -> np.ndarray:
    encoder.eval()
    chunks = []

    for x,y in tqdm(loader,leave=False):
        x = x.to(device)
        feats = encoder(x)
        feats = torch.nn.functional.normalize(feats,p=2,dim=1)
        chunks.append(feats.cpu().numpy())

    return np.concatenate(chunks,axis=0)