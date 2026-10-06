import numpy as np
import torch
from torch import nn


class BinaryAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        hidden = min(1024, max(128, 2 * latent_dim))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, input_dim)
        )

    def forward(self, x):
        continuous = self.encoder(x)
        hard = torch.where(
            continuous >= 0, torch.ones_like(continuous), -torch.ones_like(continuous)
        )
        return self.decoder(continuous + (hard - continuous).detach()), continuous


def train_compact(model, scaled, train, *, seed, epochs, batch_size, device):
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(scaled[train]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    history = []
    for _ in range(epochs):
        losses = []
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, _ = model(batch)
            loss = nn.functional.mse_loss(reconstructed, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    model.eval()
    return history
