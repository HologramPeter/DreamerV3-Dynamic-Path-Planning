from torch import nn
import torch.nn.functional as F
import torch

class DreamerV3():
    def __init__(self, embed_dim, obs_dim, action_dim, deter_dim, stoch_dim, device):
        self.encoder = Encoder(embed_dim=embed_dim, input_dim=obs_dim).to(device)
        self.rssm = RSSM(embed_dim=embed_dim, action_dim=action_dim, deter_dim=deter_dim, stoch_dim=stoch_dim).to(device)
        self.decoder = Decoder(latent_dim=deter_dim+stoch_dim, output_dim=obs_dim).to(device)
        self.device = device

    def parameters(self):
        return (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
        )

    #region Training
    def forward(self, obs, actions):
        seq_len = obs.size(0)
        embeds = []
        for t in range(seq_len):
            embeds.append(self.encoder(obs[t]))
        embeds = torch.stack(embeds, dim=0)

        # Observe RSSM states from embeddings and actions
        post_seq, prior_seq = self.observe(embeds, actions)
        post_state, post_dist = RSSMState.stack(post_seq)
        _, prior_dist = RSSMState.stack(prior_seq)

        # Decode states to reconstruct inputs
        recon = []
        for t in range(seq_len):
            recon.append(self.decoder(post_state[t]))
        recon = torch.stack(recon, dim=0)
        return recon, post_dist, prior_dist
    
    def observe(self, embeds, actions):
        # embeds (seq_len, batch_size, EMBED_DIM)
        # actions (seq_len, batch_size, 2)
        post_seq, prior_seq = [], []
        state = self.rssm.init_state(actions.size(1), self.device)
        for t in range(embeds.size(0)):
            prior = self.rssm.img_step(state, actions[t])
            state = self.rssm.obs_step(state, actions[t], embeds[t])
            prior_seq.append(prior)
            post_seq.append(state)
        return post_seq, prior_seq
    
    #endregion

    #region Running
    def attachPolicies(self, policies, reward_func, horizon=10):
        self.policies = policies
        self.reward_func = reward_func
        self.horizon = horizon

    def resetState(self):
        self.state = self.rssm.init_state(1, self.device)
        self.current_policy_horizon = 0

    def dreamPredict(self, action, obs):
        #obs shape: (obs_dim,), action shape: (action_dim,)
        obs = torch.tensor(obs, dtype=torch.float32).view(1, -1)
        action = torch.tensor(action, dtype=torch.float32).view(1, -1)

        if self.current_policy_horizon > 0:
            self.current_policy_horizon -= 1
            action = self.policies[self.current_policy_index](obs)
            # reshape (1,action_dim) to (action_dim,)
            action = action.squeeze()
            return action, {}
        
        self.current_policy_horizon = self.horizon

        info = {}
        best_index = 0
        best_reward = float('-inf')

        #encoder input shape: (1, 1, obs_dim), obs_step shape (1, 1, channel)
        self.state = self.rssm.obs_step(self.state, action, self.encoder(obs))

        for i, policy in enumerate(self.policies):
            imagined_seq = self.imagine(policy, self.horizon, self.state)
            # get accumulated reward
            reward = [self.reward_func(imagined_obs) for imagined_obs in imagined_seq]
            total_reward = sum(reward)

            if total_reward > best_reward:
                best_reward = total_reward
                best_index = i

            info[i] = {'imagined_obs_seq': imagined_seq, 'reward': total_reward}
        print(f"Best policy index: {best_index}")
        action = self.policies[best_index](obs)
        # reshape (1,action_dim) to (action_dim,)
        action = action.squeeze()

        self.current_policy_index = best_index

        return action, info

    def imagine(self, policy, horizon, state):
        seq = []
        for _ in range(horizon):
            decoded_obs = self.decoder(RSSMState.stack([state])[0])
            action = policy(decoded_obs.detach())
            action = torch.tensor(action, dtype=torch.float32).view(1, -1)
            state = self.rssm.img_step(state, action)
            seq.append(decoded_obs.detach().squeeze().numpy())
        return seq
    
    #endregion
    def save(self, path):
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'rssm_state_dict': self.rssm.state_dict(),
            'decoder_state_dict': self.decoder.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.rssm.load_state_dict(checkpoint['rssm_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])


class RSSMState:
    def __init__(self, deter, stoch, mean, std):
        self.deter = deter
        self.stoch = stoch
        self.mean = mean
        self.std = std

    def detach(self):
        return RSSMState(*[x.detach() for x in self.__dict__.values()])

    def flatten(self):
        return torch.cat([self.deter, self.stoch], dim=-1)

    def stack(rssm_states):
        return (
            torch.cat([
            torch.stack([getattr(state, attr)
                            for state in rssm_states], dim=0)
            for attr in ['deter', 'stoch']], dim=-1),
            torch.distributions.Normal(
                torch.stack([state.mean for state in rssm_states], dim=0),
                torch.stack([state.std for state in rssm_states], dim=0)
            )
        )

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

    def forward(self, x):
        # print(f"Input shape: {input_x.shape}")  # Debugging line
        return self.net(x)

class RSSM(nn.Module):
    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        #if obs not avilable, use zeros
        #input stoch state + obs + action | deter state => output deter state
        self.rnn = nn.GRUCell(stoch_dim + embed_dim + action_dim, deter_dim)

        #input deter state => output prior stoch distribution
        self.fc_prior = nn.Linear(deter_dim, 2 * stoch_dim)

        #input deter state + obs => output posterior stoch distribution
        self.fc_posterior = nn.Linear(deter_dim + embed_dim, 2 * stoch_dim)

    def init_state(self, batch_size, device):
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
            mean=torch.zeros(batch_size, self.stoch_dim, device=device),
            std=torch.ones(batch_size, self.stoch_dim, device=device)
        )

    def zero_embed(self, batch_size, device):
        return torch.zeros(batch_size, self.embed_dim, device=device)
    
    def sample_stoch(self, mean, std):
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def img_step(self, prev_state, prev_action):
        embed = self.zero_embed(prev_action.size(0), device=prev_action.device)
        return self.obs_step(prev_state, prev_action, embed)

    def obs_step(self, prev_state, prev_action, embed):
        x = torch.cat([prev_state.stoch, prev_action, embed], dim=-1)
        deter_state = self.rnn(x, prev_state.deter)

        post_x = torch.cat([deter_state, embed], dim=-1)
        post_stats = self.fc_posterior(post_x)

        post_mean, post_std = torch.chunk(post_stats, 2, dim=-1)
        post_std = F.softplus(post_std) + 0.1
        stoch_state = self.sample_stoch(post_mean, post_std)

        return RSSMState(deter_state, stoch_state, post_mean, post_std)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()  # to match original range [-1, 1]
        )
    
    def forward(self, x):
        return self.net(x)
