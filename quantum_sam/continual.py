"""Task-incremental replay and Fisher-diagonal EWC."""

from __future__ import annotations

from collections import deque
import random
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity: int = 256):
        self.data = deque(maxlen=capacity)
    def add(self, images: torch.Tensor, masks: torch.Tensor) -> None:
        self.data.extend((image.cpu(), mask.cpu()) for image, mask in zip(images, masks))
    def sample(self, count: int, device: torch.device):
        picks = random.sample(self.data, min(count, len(self.data)))
        return torch.stack([x for x, _ in picks]).to(device), torch.stack([y for _, y in picks]).to(device)
    def __len__(self): return len(self.data)


class EWC:
    """Diagonal Fisher penalty; tracks trainable parameters including circuit weights."""
    def __init__(self, model: nn.Module, strength: float = 10.0):
        self.model, self.strength, self.means, self.fisher = model, strength, {}, {}
    def penalty(self) -> torch.Tensor:
        terms = []
        for name, parameter in self.model.named_parameters():
            if name in self.fisher:
                terms.append((self.fisher[name] * (parameter - self.means[name]).pow(2)).sum())
        return self.strength * sum(terms) if terms else next(self.model.parameters()).new_zeros(())
    @torch.no_grad()
    def _snapshot(self):
        self.means = {name: p.detach().clone() for name, p in self.model.named_parameters() if p.requires_grad}
    def consolidate(self, loader, criterion, device, max_batches: int = 16):
        fisher = {name: torch.zeros_like(p) for name, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval(); batches = 0
        for images, masks in loader:
            if batches >= max_batches: break
            self.model.zero_grad(set_to_none=True)
            loss = criterion(self.model(images.to(device)), masks.to(device))
            loss.backward(); batches += 1
            for name, p in self.model.named_parameters():
                if p.grad is not None and name in fisher: fisher[name] += p.grad.detach().pow(2)
        if batches:
            self.fisher = {name: value / batches for name, value in fisher.items()}
            self._snapshot()
