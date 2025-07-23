
import torch.nn as nn
import gym
import numpy as np
import torch

class LstmWorldModel(nn.Module):
    def __init__(self, input_dim=72, output_dim=72, hidden_size=128, n=2):
        super(LstmWorldModel, self).__init__()
        # Assuming input shape is (batch_size, 3, 72) where 3 is the sequence length
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_dim * n) # Outputting 2 steps of 72 values
        self.output_dim = output_dim
        self.n = n

    def forward(self, x):
        # x shape: (batch_size, 3, 72)
        lstm_out, (hn, cn) = self.lstm(x)
        # lstm_out shape: (batch_size, 3, 128)
        # Use the output of the last time step
        last_timestep_out = lstm_out[:, -1, :] # shape: (batch_size, 128)
        output = self.linear(last_timestep_out) # shape: (batch_size, input_dim * n)
        # Reshape the output to represent 2 steps ahead
        output = output.view(output.size(0), self.n, self.output_dim) # shape: (batch_size, 2, 72)
        return output

class LstmWorldModelWrapper(gym.ObservationWrapper):
    def __init__(self, env, world_model):
        super().__init__(env)
        self.world_model = world_model
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, env.observation_space.low]),
            high=np.concatenate([env.observation_space.high, env.observation_space.high]),
            dtype=np.float32
        )

    def observation(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_dummy = torch.zeros((1, self.action_space.shape[0]))  # dummy
        pred = self.world_model(obs_tensor, action_dummy).detach().numpy()[0]
        return np.concatenate([obs, pred])