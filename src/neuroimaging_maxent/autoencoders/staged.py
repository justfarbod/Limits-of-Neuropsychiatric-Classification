from __future__ import annotations

import copy
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class IsingAutoencoder(nn.Module):
    """Symmetric staged autoencoder with a straight-through binary bottleneck."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_1: int,
        hidden_2: int,
        dropout: float,
        tanh_scale: float,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_1 = int(hidden_1)
        self.hidden_2 = int(hidden_2)
        self.dropout = float(dropout)
        self.tanh_scale = float(tanh_scale)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.LayerNorm(hidden_1),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.LayerNorm(hidden_2),
            nn.Linear(hidden_2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_2),
            nn.ReLU(),
            nn.LayerNorm(hidden_2),
            nn.Linear(hidden_2, hidden_1),
            nn.ReLU(),
            nn.LayerNorm(hidden_1),
            nn.Linear(hidden_1, input_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        use_binary_latent: bool = True,
        tanh_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.tanh_scale if tanh_scale is None else float(tanh_scale)
        z_cont = torch.tanh(scale * self.encoder(inputs))
        z_bin = torch.where(
            z_cont >= 0,
            torch.ones_like(z_cont),
            -torch.ones_like(z_cont),
        )
        latent = z_cont + (z_bin - z_cont).detach() if use_binary_latent else z_cont
        return self.decoder(latent), z_cont, z_bin


def _batch_rowwise_corr(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    x_centered = x.float() - x.float().mean(dim=1, keepdim=True)
    y_centered = y.float() - y.float().mean(dim=1, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = (
        torch.sqrt((x_centered**2).sum(dim=1) * (y_centered**2).sum(dim=1)) + eps
    )
    values = numerator / denominator
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return torch.tensor(float("nan"), device=x.device)
    return values.mean()


def _linear_schedule(epoch: int, n_epochs: int, start: float, end: float) -> float:
    if n_epochs <= 1:
        return float(end)
    return float(start + epoch / max(n_epochs - 1, 1) * (end - start))


def _run_training_phase(
    model: IsingAutoencoder,
    loader: DataLoader,
    config: Any,
    device: torch.device,
    phase_name: str,
    n_epochs: int,
    learning_rate: float,
    use_binary_latent: bool,
    tanh_start: float,
    tanh_end: float,
    noise_std: float,
    patience: int,
    use_regularization: bool,
    warmup_fraction: float,
) -> tuple[IsingAutoencoder, pd.DataFrame]:
    ae = config.autoencoder
    input_mode = config.preprocessing.fc_input_mode
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_scale = float(tanh_start)
    patience_counter = 0
    history: list[dict[str, Any]] = []
    for epoch in range(n_epochs):
        model.train()
        scale = _linear_schedule(epoch, n_epochs, tanh_start, tanh_end)
        model.tanh_scale = scale
        warmup_epochs = (
            int(round(warmup_fraction * n_epochs)) if use_regularization else 0
        )
        if not use_regularization or epoch < warmup_epochs:
            regularization_progress = 0.0
        else:
            regularization_progress = (epoch - warmup_epochs) / max(
                n_epochs - warmup_epochs - 1, 1
            )
        totals = {
            key: 0.0
            for key in (
                "loss",
                "objective",
                "reconstruction",
                "correlation",
                "binary",
                "balance",
                "separation",
                "bit_accuracy",
                "row_correlation",
                "abs_z",
            )
        }
        n_batches = 0
        for (clean,) in loader:
            clean = clean.to(device, non_blocking=True)
            noisy = (
                clean + noise_std * torch.randn_like(clean) if noise_std > 0 else clean
            )
            output, z_cont, _ = model(
                noisy,
                use_binary_latent=use_binary_latent,
                tanh_scale=scale,
            )
            if input_mode == "binary":
                reconstruction = torch.sigmoid(output)
                reconstruction_loss = F.binary_cross_entropy_with_logits(output, clean)
            else:
                reconstruction = output
                reconstruction_loss = F.mse_loss(output, clean)
            use_corr = (
                ae.use_correlation_loss
                and input_mode == "continuous"
                and ae.corr_loss_weight > 0
            )
            if use_corr:
                correlation = _batch_rowwise_corr(clean, reconstruction)
                correlation_loss = (
                    1.0 - correlation
                    if torch.isfinite(correlation)
                    else torch.tensor(0.0, device=device)
                )
            else:
                correlation_loss = torch.tensor(0.0, device=device)
            objective = reconstruction_loss + ae.corr_loss_weight * correlation_loss
            binary_loss = ((z_cont.abs() - 1) ** 2).mean()
            balance = ((z_cont.mean(dim=0)) ** 2).sum()
            separation = torch.exp(-5 * z_cont.abs()).mean()
            loss = objective
            if use_regularization:
                loss = loss + (
                    ae.bin_weight_max * regularization_progress * binary_loss
                    + ae.bal_weight_max * regularization_progress * balance
                    + ae.sep_weight * regularization_progress * separation
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                row_correlation = _batch_rowwise_corr(clean, reconstruction)
                if input_mode == "binary":
                    bit_accuracy = (
                        ((reconstruction >= 0.5) == clean.bool()).float().mean()
                    )
                else:
                    bit_accuracy = torch.tensor(float("nan"), device=device)
            totals["loss"] += float(loss.item())
            totals["objective"] += float(objective.item())
            totals["reconstruction"] += float(reconstruction_loss.item())
            totals["correlation"] += float(correlation_loss.item())
            totals["binary"] += float(binary_loss.item())
            totals["balance"] += float(balance.item())
            totals["separation"] += float(separation.item())
            totals["bit_accuracy"] += (
                float(bit_accuracy.item()) if torch.isfinite(bit_accuracy) else 0.0
            )
            totals["row_correlation"] += (
                float(row_correlation.item())
                if torch.isfinite(row_correlation)
                else 0.0
            )
            totals["abs_z"] += float(z_cont.abs().mean().item())
            n_batches += 1
        average = {key: value / max(n_batches, 1) for key, value in totals.items()}
        history.append(
            {
                "phase": phase_name,
                "epoch": epoch,
                "global_epoch": None,
                "loss": average["loss"],
                "objective_loss": average["objective"],
                "reconstruction_loss": average["reconstruction"],
                "correlation_loss": average["correlation"],
                "bit_accuracy": (
                    average["bit_accuracy"] if input_mode == "binary" else np.nan
                ),
                "batch_edge_r": average["row_correlation"],
                "mean_abs_z": average["abs_z"],
                "bin_loss": average["binary"],
                "balance_loss": average["balance"],
                "separation_loss": average["separation"],
                "reg_progress": regularization_progress,
                "tanh_scale": scale,
                "noise_std": noise_std,
                "use_binary_latent": use_binary_latent,
                "fc_input_mode": input_mode,
            }
        )
        selection_score = average["objective"]
        if use_regularization:
            selection_score += 0.002 * average["binary"] + 0.001 * average["balance"]
        if selection_score < best_score:
            best_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            best_scale = scale
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.tanh_scale = best_scale
    return model, pd.DataFrame(history)


def train_autoencoder(
    model: IsingAutoencoder,
    loader: DataLoader,
    config: Any,
    device: torch.device,
) -> tuple[IsingAutoencoder, pd.DataFrame]:
    """Run soft pretraining followed by binary fine-tuning."""
    ae = config.autoencoder
    input_mode = config.preprocessing.fc_input_mode
    pretrain_noise = ae.pretrain_noise_std if input_mode == "continuous" else 0.0
    finetune_noise = ae.finetune_noise_std if input_mode == "continuous" else 0.0
    model, pretrain = _run_training_phase(
        model,
        loader,
        config,
        device,
        "Pretrain",
        ae.pretrain_epochs,
        ae.pretrain_lr,
        False,
        ae.pretrain_tanh_start,
        ae.pretrain_tanh_end,
        pretrain_noise,
        ae.pretrain_patience,
        False,
        0.0,
    )
    model, finetune = _run_training_phase(
        model,
        loader,
        config,
        device,
        "FineTune",
        ae.finetune_epochs,
        ae.finetune_lr,
        True,
        ae.finetune_tanh_start,
        ae.finetune_tanh_end,
        finetune_noise,
        ae.finetune_patience,
        True,
        ae.finetune_reg_warmup_fraction,
    )
    history = pd.concat([pretrain, finetune], ignore_index=True)
    history["global_epoch"] = np.arange(len(history))
    return model, history


def extract_latents(
    model: IsingAutoencoder,
    prepared_vectors: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        torch.from_numpy(np.asarray(prepared_vectors, dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    continuous: list[np.ndarray] = []
    binary_pm1: list[np.ndarray] = []
    model.to(device).eval()
    with torch.no_grad():
        for values in loader:
            _, z_cont, z_bin = model(values.to(device, non_blocking=True))
            continuous.append(z_cont.cpu().numpy())
            binary_pm1.append(z_bin.cpu().numpy())
    z_cont = np.vstack(continuous).astype(np.float32)
    z_pm1 = np.vstack(binary_pm1).astype(np.float32)
    z_01 = ((z_pm1 + 1.0) / 2.0).astype(np.float32)
    return z_cont, z_pm1, z_01
