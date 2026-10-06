import torch
from torch import nn
import numpy as np


class MaxEnt(nn.Module):
    """
    Maximum Entropy (Ising-like) model for binary systems.
    Supports exact computation of probabilities and moments
    via full state enumeration.
    """

    def __init__(self, n, device="cuda"):
        super().__init__()
        if not 1 <= n <= 22:
            raise ValueError("Full enumeration requires 1 to 22 variables.")
        self.n = n
        self.device = device
        self.h = nn.Parameter(0.1 * torch.randn(n, device=device))
        self.J = nn.Parameter(0.1 * torch.randn(n, n, device=device))
        self.states = self._get_all_states()

    def _symmetrize_J(self):
        """Enforce symmetry and zero out diagonal of interaction matrix."""
        J_sym = (self.J + self.J.T) / 2
        J_sym = J_sym - torch.diag(torch.diag(J_sym))
        return J_sym

    def _get_all_states(self):
        """Enumerate all 2^n binary configurations as a (2^n x n) tensor."""
        codes = np.arange(2**self.n, dtype=np.uint32)
        states = np.empty((len(codes), self.n), dtype=np.float32)
        for column in range(self.n):
            states[:, column] = (codes >> (self.n - column - 1)) & 1
        return torch.as_tensor(states, dtype=torch.float32, device=self.device)

    def _energy(self, states=None):
        """Compute Ising energy: lower energy ⇒ higher probability."""
        if states is None:
            states = self.states
        J = self._symmetrize_J()
        linear = states @ self.h
        quadratic = torch.einsum("bi,ij,bj->b", states, J, states)
        return -linear - 0.5 * quadratic

    def _compute_probabilities(self):
        """Compute normalized probabilities for all binary states."""
        E = self._energy()
        log_Z = torch.logsumexp(-E, dim=0)
        probs = torch.exp(-E - log_Z)
        assert torch.isclose(probs.sum(), torch.tensor(1.0, device=self.device))
        return probs, torch.exp(log_Z)

    def fit(
        self, data_np, lr=1e-2, steps=10000, verbose=True, patience=100, lambda_=0.0
    ):
        """
        Fit the MaxEnt model by minimizing negative log-likelihood (NLL)
        with optional L1 regularization and early stopping.
        """
        data = torch.tensor(data_np, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_loss, patience_counter = None, 0
        best_model = {"h": self.h.clone(), "J": self._symmetrize_J().clone()}

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.loss(data, lambda_=lambda_)
            loss.backward()
            optimizer.step()

            if verbose:
                print(f"Step {step:5d}: Loss = {loss.item():.10f}", end="\r")

            current_loss = loss.item()
            if best_loss is None or current_loss < best_loss:
                best_loss, patience_counter = current_loss, 0
                best_model["h"] = self.h.clone()
                best_model["J"] = self._symmetrize_J().clone()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(
                            f"\nEarly stopping at step {step}. Best NLL: {best_loss:.5f}"
                        )
                    break

        self.h.data = best_model["h"]
        self.J.data = best_model["J"]
        self._compute_probabilities()

    def get_model_marginals(self):
        probs, _ = self._compute_probabilities()
        first_moment = torch.einsum("b,bi->i", probs, self.states)
        second_moment = torch.einsum("b,bi,bj->ij", probs, self.states, self.states)

        mask = torch.triu(
            torch.ones(self.n, self.n, device=self.device), diagonal=1
        ).bool()

        return first_moment, second_moment[mask]

    def get_empirical_marginals(self, data_np):
        """Compute empirical means and correlations from data."""
        data = (
            torch.tensor(data_np, dtype=torch.float32, device=self.device)
            if not isinstance(data_np, torch.Tensor)
            else data_np
        )
        first_moment = data.mean(dim=0)
        second_moment = (data.T @ data) / data.size(0)
        mask = (torch.triu(torch.ones(self.n, self.n), diagonal=1) == 1).reshape(
            self.n, self.n
        )
        return first_moment, torch.triu(second_moment)[mask]

    def log_prob(self, x):
        """
        Compute log probability of given binary states.
        Accepts x of shape (n,) or (B, n).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        _, Z = self._compute_probabilities()
        E = self._energy(x)
        return -E - torch.log(Z)

    def loss(self, x, lambda_=0.0):
        """Negative log-likelihood loss with L1 regularization on h and J."""
        x = (
            torch.tensor(x, dtype=torch.float32, device=self.device)
            if not isinstance(x, torch.Tensor)
            else x
        )
        log_probs = self.log_prob(x).mean()
        l1_penalty = lambda_ * (
            self.h.abs().sum() + 0.5 * self._symmetrize_J().abs().sum()
        )
        return -log_probs + l1_penalty
