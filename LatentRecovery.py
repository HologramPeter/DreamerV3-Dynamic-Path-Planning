import torch
import torch.nn as nn

class LatentRecovery(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + latent_dim + 1, hidden[0]),  # +1 for delta
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden[1], latent_dim),
        )

    def forward(self, obs_t_delta, latent_t, delta_t):
        x = torch.cat([obs_t_delta, latent_t, delta_t], dim=-1)
        correction = self.mlp(x)
        return correction
