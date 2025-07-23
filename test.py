import torch
# Input: sequence of observations
obs = env.step(prev_action) #360 + 4
actions = TD3.select_action(obs) #2

encoder = Encoder()
rssm = RSSM()
decoder = Decoder()

optimizer = some_optimizer
kl_divergence = some_function

embedded = encoder(obs[:, 4:]) #360 -> embed_size

#concatenate embedded with skipped connections
embedded = torch.cat([embedded, obs[:, :4]], dim=-1) #embed_size + 4

#sequences
posterior, prior = rssm.observe(embedded, actions) #stoch_dim+deter_dim
reconstructed = decoder(posterior) #stoch_dim+deter_dim -> 360

#use rssm.img_step for imagination
# post, prior = rssm.img_step(prev_state, prev_action, embedded_observation)

recon_loss = mse(reconstructed, obs)
kl_loss = kl_divergence(posterior, prior).sum()

#define beta
beta = 0.5
model_loss = recon_loss + beta * kl_loss

optimizer.zero_grad()
model_loss.backward()# updates encoder, RSSM, decoder
optimizer.step()