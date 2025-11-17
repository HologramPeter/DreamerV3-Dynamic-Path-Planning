import torch
import torch.nn as nn

class LatentRecovery(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + latent_dim + 1, hidden),  # +1 for delta
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
            nn.Dropout(p=0.2)
        )

    def forward(self, obs_t_delta, latent_t, delta_t):
        # delta_t: tensor of shape [batch, 1] or [batch]
        delta_t = delta_t.unsqueeze(-1) if delta_t.dim() == 1 else delta_t
        x = torch.cat([obs_t_delta, latent_t, delta_t], dim=-1)
        correction = self.mlp(x)
        return latent_t + correction
