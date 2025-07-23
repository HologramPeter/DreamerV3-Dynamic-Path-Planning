from DreamerV3 import DreamerV3
from WebotsGymAddon import DataSampler, generateFileName
import numpy as np
import matplotlib.pyplot as plt
import torch

SEED = 42

def train(model, sampler, optimizer, steps=1000):
    reconstruction_loss_weight = torch.tensor([
        1, 1,  # linear and angular velocity
        1, 1,  # robot x and y position
        1, 1,  # sin and cos of heading diff
        2, 2, 2, 2,  # lidar readings
        2, 2, 2, 2   # lidar readings
    ])

    def loss_fn(recon, obs, post_dist, prior_dist):
        # Reconstruction loss (MSE)
        mse = (((recon - obs) ** 2)).mean()
        recon_loss = mse * reconstruction_loss_weight
        recon_loss = recon_loss.sum()

        # KL divergence loss
        kl_loss = torch.distributions.kl.kl_divergence(
                post_dist, prior_dist
            ).mean()
        
        return (recon_loss, kl_loss)

    history = []
    for _ in range(steps):
        actions, obs = sampler.sample()
        recon, post_dist, prior_dist = model.forward(obs, actions)

        recon_loss, kl_loss = loss_fn(recon, obs, post_dist, prior_dist)
        loss = recon_loss + kl_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #validation
        val_actions, val_obs = sampler.sample(train=False)
        val_recon, val_post_dist, val_prior_dist = dreamer_v3.forward(val_obs, val_actions)
        val_recon_loss, val_kl_loss = loss_fn(val_recon, val_obs, val_post_dist, val_prior_dist)
        history.append((recon_loss.item(), kl_loss.item(), val_recon_loss.item(), val_kl_loss.item()))

    return history, dreamer_v3


def join_data_list(data_list):
    offset = 0
    adjusted = []
    for arr in data_list:
        arr[:, 0] += offset
        offset = arr[:, 0].max()
        adjusted.append(arr)
    return np.concatenate(adjusted, axis=0)
    

if __name__ == "__main__":
    device = torch.device("cpu")

    train_steps = 500
    batch_size = 5
    seq_len = 50

    data_list = []
    for data_path in [
        'data/env_data_right_turn.csv',
        'data/env_data_left_turn.csv'
    ]:
        data_list.append(np.loadtxt(data_path, delimiter=','))

    sampler = DataSampler(join_data_list(data_list), batch_size, seq_len)
    sampler.seed(SEED)

    dreamer_v3 = DreamerV3(
        embed_dim=32,
        obs_dim=sampler.obs_dim(),
        action_dim=2,
        deter_dim=128,
        stoch_dim=32,
        device=device)

    optimizer = torch.optim.Adam(
        dreamer_v3.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-8
    )

    history, dreamer_v3 = train(dreamer_v3, sampler, optimizer, steps=train_steps)

    # Save the model
    name = generateFileName(prefix='dreamer_v3_model')
    dreamer_v3.save(name + ".pth")
    # #try to load the model
    # dreamer_v3 = dreamer_v3.load(name + ".pth")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in history])
    plt.plot([x[2] for x in history])
    plt.xlabel("Training Steps")
    plt.ylabel("Reconstruction Loss")
    plt.title("Reconstruction Loss Over Time")
    plt.legend(['Train Recon Loss', 'Val Recon Loss'])

    plt.subplot(1, 2, 2)
    plt.plot([x[1] for x in history])
    plt.plot([x[3] for x in history])
    plt.xlabel("Training Steps")
    plt.ylabel("KL Divergence Loss")
    plt.title("KL Divergence Loss Over Time")
    plt.legend(['Train KL Loss', 'Val KL Loss'])

    plt.tight_layout()
    plt.show()
