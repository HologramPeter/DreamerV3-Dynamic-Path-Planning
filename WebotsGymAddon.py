from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import datetime

def generateFileName(prefix):
    return f"{prefix}-{datetime.datetime.now().strftime('%m%d_%H%M')}.zip"

class NullCallback(BaseCallback):
    def __init__(self, record_path, verbose=False):
        super().__init__(verbose)

    def _verbose(self, msg):
        return

    def _on_step(self) -> bool:
        return True
    
    def nextRollout(self):
        return
    
    def close(self):
        return
    
    def save(self, *args):
        return

class SaveObsCallback(BaseCallback):
    def __init__(self, record_path, verbose=False):
        super().__init__(verbose)
        self.file = open(record_path, 'a')
        self.seq = np.array([[0]])
        self.rollout = np.array([[0]])

    def _verbose(self, msg):
        if self.verbose:
            print("[RECORD]"+msg)

    def _on_step(self) -> bool:
        self.save(
            action = self.locals.get("actions"),
            new_obs = self.locals.get("new_obs"),
            reward = self.locals.get("rewards"),
            terminated = self.locals.get("terminated"),
            truncated = self.locals.get("truncated"),
            info = self.locals.get("infos"))
        return True

    def save(self, action, new_obs, reward, terminated, truncated, info):
        np.savetxt(self.file, 
                   np.concatenate((self.rollout, self.seq, action, new_obs), axis=1),
                   delimiter=',',
                   fmt='%.6f')
        
        self._verbose(f"Rollout {self.rollout}, Sequence {self.seq}")
        self.seq[0][0] += 1
        if terminated or truncated:
            self.file.flush()
            self.rollout[0][0] += 1
            self.seq[0][0] = 0
            self._verbose(f"Finished Rollout {self.rollout}")
    
    def nextRollout(self):
        self.file.flush()
        self.rollout[0][0] += 1
        self.seq[0][0] = 0
        self._verbose(f"Next Rollout {self.rollout}")
    
    def close(self):
        self.file.close()
        self._verbose("File closed")


class DataSampler:
    def __init__(self, data, batch_size, seq_count):
        self.data = data
        self.batch_size = batch_size
        self.seq_count = seq_count
        self.data_channels = self.data.shape[1] - 2  # Exclude rollout and sequence numbers and actions

        rollout_ids = self.data[:, 0].astype(int)
        self.unique_ids = np.unique(rollout_ids)
        self.train_map = {}
        self.val_map = {}

        # Create a mapping from rollout ID to indices in the data
        # separate into training and validation sets
        for rid in self.unique_ids:
            if np.sum(rollout_ids == rid) >= seq_count:
                if rid % 5 == 0:
                    self.val_map[rid] = np.where(rollout_ids == rid)[0]
                else:
                    self.train_map[rid] = np.where(rollout_ids == rid)[0]

    def seed(self, seed):
        np.random.seed(seed)

    def obs_dim(self):
        return self.data_channels - 2  # Exclude action channels

    def sample(self, train=True):
        n = self.batch_size
        s = self.seq_count

        if train:
            rollouts = np.random.choice(list(self.train_map.keys()), n, replace=True)
        else:
            rollouts = np.random.choice(list(self.val_map.keys()), n, replace=True)

        sampled = np.zeros((s, n, self.data_channels), dtype=np.float32)

        for batch_idx, rollout_id in enumerate(rollouts):
            indices = self.train_map[rollout_id] if train else self.val_map[rollout_id]
            max_start = len(indices) - s
            start_pos = np.random.randint(0, max_start + 1)
            chosen_indices = indices[start_pos : start_pos + s]
            sampled[:, batch_idx, :] = self.data[chosen_indices, 2:]

        actions = sampled[:, :, :2]
        obs = sampled[:, :, 2:]
        return actions, obs
    

class TensorPolicyWrapper:
    class LambdaWrapper:
        def __init__(self, func):
            self.func = func

        def predict(self, obs):
            return (self.func(obs), None)

    def initWithLambda(func):
        return TensorPolicyWrapper(TensorPolicyWrapper.LambdaWrapper(func))

    def __init__(self, agent):
        self.agent = agent

    def __call__(self, obs):
        #reshape (1, 1,obs_dim) to (obs_dim,)
        input_obs = obs.squeeze()
        actions, _states = self.agent.predict(input_obs)
        #reshape (action_dim,) to (1,1,action_dim)
        output_actions = actions.reshape(1, 1, -1)
        return output_actions
    

import random

def custom_gap_sampler(latent_seq, obs_seq, max_gap=30):
    """
    latent_seq: Tensor [batch, T, latent_dim]
    obs_seq: Tensor [batch, T, obs_dim]
    max_gap: maximum allowed delta

    Yields tuples (latent_t, obs_t_delta, delta_t)
    """
    batch_size, T, _ = latent_seq.shape

    for b in range(batch_size):
        t = 0
        while t < T - 1:
            # randomly choose gap length
            delta = random.randint(1, min(max_gap, T - t - 1))
            latent_t = latent_seq[b, t, :]
            obs_t_delta = obs_seq[b, t + delta, :]
            yield latent_t, obs_t_delta, delta
            t += 1  # or t += delta if you want non-overlapping samples
