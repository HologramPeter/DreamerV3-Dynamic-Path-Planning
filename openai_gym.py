
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env

from WebotsGymEnvironment import WebotsGymEnvironment
from WebotsGymAddon import TensorPolicyWrapper, SaveObsCallback, generateFileName
from WebotsReward import *

import torch

from DreamerV3 import DreamerV3

from stable_baselines3.common.noise import NormalActionNoise

DATA_FOLDER = 'data/'
LOG_FILE_PATH = DATA_FOLDER + 'webots_gym.log'
SEED = 42

LOG_FILE = open(LOG_FILE_PATH, 'a')
LOG = lambda *arg: print(*arg, file=LOG_FILE)

env = WebotsGymEnvironment(max_episode_steps=1000)
# check_env(env)
env.seed(SEED)


#record info the environment as a csv file
def runRandomEnv(env, steps, repeat_action, callback=None):
        obs = env.reset()
        for _ in range(steps // repeat_action):
            action = env.action_space.sample()
            for _ in range(repeat_action):
                obs, reward, terminated, truncated, info = env.step(action)
                if callback:
                    callback.save(np.array(action), obs, reward, terminated, truncated, info)
                if terminated or truncated:
                    env.reset()

def runWithAgent(env, agent, runs, callback=None):
    obs, info = env.reset()

    run_info = [] #(total_reward, steps, success)
    accum_reward = 0
    steps = 0
    while runs > 0:
        action, _states = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        accum_reward += reward
        steps += 1

        if callback:
            callback.save(action.reshape((1, -1)), obs.reshape((1, -1)), reward, terminated, truncated, info)
        if terminated or truncated:
            run_info.append((accum_reward, steps, info.get('success', False)))
            obs, info = env.reset()
            accum_reward = 0
            steps = 0
            runs -= 1
    return run_info

def runWithDreamer(env, agents, dreamer, horizon, runs,
                   reward_generator, callback=None):
    policies = [TensorPolicyWrapper(agent) for agent in agents]
    # policies.append(TensorPolicyWrapper.initWithLambda(
    #     lambda obs: np.zeros(env.action_space.shape, dtype=np.float32)))  # No-op policy
    # policies.append(TensorPolicyWrapper.initWithLambda(
    #     lambda obs: np.ones(env.action_space.shape, dtype=np.float32)))  # full forward policy
    # policies.append(TensorPolicyWrapper.initWithLambda(
    #     lambda obs: -np.ones(env.action_space.shape, dtype=np.float32)))  # full reverse policy
    dreamer.attachPolicies(policies, reward_generator, horizon)

    dreamer.resetState()
    obs, info = env.reset()

    action = np.zeros(env.action_space.shape)
    run_info = [] #(total_reward, steps, success)
    accum_reward = 0
    steps = 0
    while runs > 0:
        #predict horizon observation using dreamer
        action, dream_info = dreamer.dreamPredict(action, obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # pos = obs[2:4] * env.bounds_threshold
        # nodes = [(pos[0], pos[1], 0.1), (0, 0, 0.1)]  # green line to goal
        # env.drawLines(green=nodes)

        green_index = dreamer.current_policy_index # 0 or 1
        red_index = 1-green_index
        # draw lines imagined_obs_seq
        imagined_obs_seq_green = dream_info.get(green_index, {}).get('imagined_obs_seq', [])
        imagined_obs_seq_red = dream_info.get(red_index, {}).get('imagined_obs_seq', [])
        if len(imagined_obs_seq_red) > 0:
            print(imagined_obs_seq_red[0][2:4])
            print(imagined_obs_seq_red[0][2:4]*env.bounds_threshold)
            green = [(obs[2]*env.bounds_threshold, obs[3]*env.bounds_threshold, 0.1) for obs in imagined_obs_seq_green]
            red = [(obs[2]*env.bounds_threshold, obs[3]*env.bounds_threshold, 0.1) for obs in imagined_obs_seq_red]
            env.drawLines(green=green, red=red)

        accum_reward += reward
        steps += 1

        if callback:
            callback.save(action.reshape((1, -1)), obs.reshape((1, -1)), reward, terminated, truncated, info)
        if terminated or truncated:
            run_info.append((accum_reward, steps, info.get('success', False)))
            dreamer.resetState()
            obs, info = env.reset()

            accum_reward = 0
            steps = 0
            runs -= 1
    return run_info

def main(mode, models, name,
         reward_generator,
         record_path, record_verbose,
         dreamer_path=None):

    callback = SaveObsCallback(record_path, verbose=record_verbose)

    env.setRewardFunction(reward_generator)

    ######################################

    if mode == RECORD:
        runRandomEnv(env, steps=1000, repeat_action=10, callback=callback)

    elif mode == TRAIN:
        if models:
            agent = TD3.load(models, env=env)
            print(f"Model loaded from {models}")
        else:
            policy_kwargs = dict(
                net_arch=[400, 300],
                # activation_fn=torch.nn.ReLU(),
            )

            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

            agent = TD3('MlpPolicy', env, policy_kwargs=policy_kwargs,
                        train_freq=1, gradient_steps=1,
                        action_noise=action_noise
                    )
            print("New agent created")
            
        agent.learn(total_timesteps=5e4, callback=callback)

        print('Training is finished, press `Y` to save and replay...')
        # env.wait_keyboard()

        name = generateFileName(prefix='td3_model3_' + name)
        agent.save(DATA_FOLDER + name)
        print(f"Model saved as {name}.zip")
        return
        runWithAgent(env, agent, callback=callback)

    elif mode == REPLAY:
        for agent_path in models:
            agent = TD3.load(agent_path, env=env)
            print(f"Model loaded from {agent_path}")

            run_info_list = runWithAgent(env, agent, runs=100, callback=callback)
            for i, run_info in enumerate(run_info_list):
                total_reward, steps, success = run_info
                LOG(f"{agent_path}, {i}, {total_reward/steps}, {steps}, {success}")

    elif mode == DREAMER:
        if not dreamer_path or len(models) == 0:
            raise ValueError("Model path must be provided for Dreamer mode.")

        dreamer_v3 = DreamerV3(
            embed_dim=32,
            obs_dim=14,
            action_dim=2,
            deter_dim=128,
            stoch_dim=32,
            device=torch.device("cpu")
        )

        # dreamer_v3.load(dreamer_path)
        # print(f"Dreamer model loaded from {dreamer_path}")
        
        agents = []
        for agent_path in models:
            agent = TD3.load(agent_path, env=env)
            print(f"Agent model loaded from {agent_path}")
            agents.append(agent)

        LOG(f"Running Dreamer with {len(agents)} agents...")
        for horizon in [30]:
            print(f"Running Dreamer with horizon {horizon}...")
            run_info_list = runWithDreamer(env, agents, dreamer_v3, horizon=horizon,
                                           runs=100,
                                           reward_generator=reward_generator,
                                           callback=callback)

            for i, run_info in enumerate(run_info_list):
                total_reward, steps, success = run_info
                LOG(f"{horizon}, {i}, {total_reward/steps}, {steps}, {success}")

if __name__ == '__main__':
    TRAIN = 0
    REPLAY = 1
    RECORD = 2
    DREAMER = 3

    # record_path = 'env_data_dreamer_6x6.csv'
    record_path = 'TEST.csv'
    model_path = 'td3_model3_straight2_0723_0933.zip'
    dreamer_path = 'dreamer_1/dreamer_v3_model_0723_0921.zip.pth'

    models = [
        'dreamer_1/td3_model3_left_0723_0030.zip',
        'dreamer_1/td3_model3_right_0722_2359.zip',
        # 'dreamer_1/td3_model3_straight_0723_1153.zip',
        # 'dreamer_1/td3_model3_straight2_0723_1126.zip',
    ]

    model_paths = [DATA_FOLDER + model for model in models]
    dreamer_path = DATA_FOLDER + dreamer_path
    model_path = DATA_FOLDER + model_path
    record_path = DATA_FOLDER + record_path


    record_path2 = 'env_data_straight2_6x6.csv'
    record_path2 = DATA_FOLDER + record_path2
    
    
    main(
        # RECORD,

        # TRAIN,
        # models=None,
        # # reward_generator=RewardGeneratorRightTurn(verbose=True),
        # # reward_generator=RewardGeneratorLeftTurn(verbose=True),
        # reward_generator=RewardGeneratorStraight(verbose=True),

        # REPLAY,
        # models=model_paths,
        # reward_generator=RewardGeneratorSteering(verbose=True),

        DREAMER,
        dreamer_path=dreamer_path,
        models=model_paths,
        reward_generator=RewardGeneratorSteering(verbose=False),

        name='test',
        record_path=record_path,
        record_verbose=False)
    
    # main(
    #     # RECORD,

    #     # TRAIN,
    #     # models=None,
    #     # # reward_generator=RewardGeneratorRightTurn(verbose=True),
    #     # # reward_generator=RewardGeneratorLeftTurn(verbose=True),
    #     # reward_generator=RewardGeneratorStraight(verbose=True),

    #     REPLAY,
    #     models=model_paths,
    #     reward_generator=RewardGeneratorSteering(verbose=True),

    #     # DREAMER,
    #     # dreamer_path=dreamer_path,
    #     # models=model_paths,
    #     # reward_generator=RewardGeneratorSteering(verbose=False),

    #     name='test',
    #     record_path=record_path,
    #     record_verbose=False)
    
    
