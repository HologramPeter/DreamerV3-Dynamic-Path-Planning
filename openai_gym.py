
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env

from WebotsGymEnvironment import WebotsGymEnvironment
from WebotsGymAddon import TensorPolicyWrapper, SaveObsCallback, generateFileName
from WebotsReward import *

import torch

from DreamerV3 import DreamerV3

from stable_baselines3.common.noise import NormalActionNoise

DATA_FOLDER = 'data_wk9/'
LOG_FILE_PATH = DATA_FOLDER + 'webots_gym.log'
# 982040
# Rand = np.random.randint(0, 1000000)
# SEED = Rand
# SEED = 982040
# print(f"Random seed: {SEED}")
SEED = 42

LOG_FILE = open(LOG_FILE_PATH, 'a')
LOG = lambda *arg: print(*arg, file=LOG_FILE)
def FLUSH_LOG():
    LOG_FILE.flush()

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



    # simple polcies

    # def TurnToGoalPolicy(obs):
    #     #while cos heading is positive, turn right, otherwise turn left
    #     sin_heading = obs[4]  # sin_heading is at index 4
    #     cos_heading = obs[5]  # cos_heading is at index 5
    #     #turn heading to 0 base on sin and cos heading
    #     if cos_heading > 0.9:
    #         #no need to turn full forward
    #         return np.array([1.0, 1.0], dtype=np.float32)
    #     else:
    #         #turn left or right based on sin heading
    #         if sin_heading > 0:
    #             return np.array([-1.0, 1.0], dtype=np.float32)
    #         else:
    #             return np.array([1.0, -1.0], dtype=np.float32)

    # policies.append(TensorPolicyWrapper.initWithLambda(TurnToGoalPolicy))

    # policies = []
    # policies.append(TensorPolicyWrapper.initWithLambda(
    #     lambda obs: np.zeros(env.action_space.shape, dtype=np.float32)))  # No-op policy
    # policies.append(TensorPolicyWrapper.initWithLambda(
    #     lambda obs: np.ones(env.action_space.shape, dtype=np.float32)))  # full forward policy
    # policies.append(TensorPolicyWrapper.initWithLambda(
    #     lambda obs: -np.ones(env.action_space.shape, dtype=np.float32)))  # full reverse policy
    # policies.append(TensorPolicyWrapper.initWithLambda(
    #     lambda obs: np.array([1, -1], dtype=np.float32)))
    # policies.append(TensorPolicyWrapper.initWithLambda(
    #     lambda obs: np.array([-1, 1], dtype=np.float32)))

    dreamer.attachPolicies(policies, reward_generator, horizon)
    dreamer.setInferredSettings(
        step_size=0.032,
        wheel_base=0.16,
        wheel_radius=0.033,
        max_wheel_speed=6.67,
    )

    dreamer.resetState()
    obs, info = env.reset()

    action = np.zeros(env.action_space.shape)
    run_info = [] #(total_reward, steps, success)
    accum_reward = 0
    steps = 0
    while runs > 0:
        #predict horizon observation using dreamer
        action, dream_info = dreamer.dreamPredict(action, obs, info.get('heading', 0.0))
        obs, reward, terminated, truncated, info = env.step(action)

        green_index = dreamer.current_policy_index
        if len(dream_info) > 0:
            lines = []
            for policy_index, policy_info in dream_info.items():
                if policy_index == green_index:
                    green = [(_obs[2]*env.bounds_threshold, _obs[3]*env.bounds_threshold, 0.1) for _obs in policy_info.get('imagined_obs_seq', [])]
                else:
                    lines.append([(_obs[2]*env.bounds_threshold, _obs[3]*env.bounds_threshold, 0.1) for _obs in policy_info.get('imagined_obs_seq', [])])
            env.drawLines(green=green, reds=lines)

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
         dreamer_path=None,
         train_steps=1e4):

    callback = SaveObsCallback(record_path, verbose=record_verbose)

    env.setRewardFunction(reward_generator)

    ######################################

    if mode == RECORD:
        runRandomEnv(env, steps=1000, repeat_action=10, callback=callback)

    elif mode == TRAIN:
        if models:
            agent = TD3.load(models[0], env=env)
            print(f"Model loaded from {models[0]}")
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
            
        agent.learn(total_timesteps=train_steps, callback=callback)

        name = generateFileName(prefix='td3_model3_' + name)
        agent.save(DATA_FOLDER + name)
        print(f"Model saved as {name}")
        return name

    elif mode == REPLAY:
        for agent_path in models:
            agent = TD3.load(agent_path, env=env)
            print(f"Model loaded from {agent_path}")

            run_info_list = runWithAgent(env, agent, runs=100, callback=callback)
            for i, run_info in enumerate(run_info_list):
                total_reward, steps, success = run_info
                LOG(f"{agent_path},{i},{total_reward/steps},{steps},{1 if success else 0}")
            FLUSH_LOG()

    elif mode == DREAMER:
        if not dreamer_path or len(models) == 0:
            raise ValueError("Model path must be provided for Dreamer mode.")

        dreamer_v3 = DreamerV3(
            embed_dim=64,
            obs_dim=14,
            action_dim=2,
            deter_dim=128,
            stoch_dim=32,
            device=torch.device("cpu")
        )

        dreamer_v3.load(dreamer_path)
        print(f"Dreamer model loaded from {dreamer_path}")
        
        agents = []
        for agent_path in models:
            agent = TD3.load(agent_path, env=env)
            print(f"Agent model loaded from {agent_path}")
            agents.append(agent)

        print(f"Running Dreamer with {len(agents)} agents...")
        for horizon in [10,20]:
            print(f"Running Dreamer with horizon {horizon}...")
            run_info_list = runWithDreamer(env, agents, dreamer_v3, horizon=horizon,
                                           runs=100,
                                           reward_generator=reward_generator,
                                           callback=callback)
            for i, run_info in enumerate(run_info_list):
                total_reward, steps, success = run_info
                LOG(f"{horizon},{i},{total_reward/steps},{steps},{1 if success else 0}")
            FLUSH_LOG()

if __name__ == '__main__':
    TRAIN = 0
    REPLAY = 1
    RECORD = 2
    DREAMER = 3

    record_path = 'env_data_dreamer_6x6.csv'
    model_path = 'td3_model3_straight2_0723_0933.zip'
    # dreamer_path = 'dreamer_1/dreamer_v3_model2_0723_1406.zip.pth'
    # dreamer_path = 'dreamer_v3_model_0728_0746.zip.pth' # week 9
    dreamer_path = 'dreamer_v3_model_week9_0731_0111.zip.pth' # week 9

    models = [
        'td3_model3_LeftTurn_stage2_0731_1009.zip',
        'td3_model3_RightTurn_stage2_0731_0947.zip',
        'td3_model3_Steering_stage2_0731_1031.zip'
    ]

    model_paths = [DATA_FOLDER + model for model in models]

    # model_paths.append('data/dreamer_1/td3_model3_right_turn_0720_0040.zip')
    # model_paths.append('data/dreamer_1/td3_model3_left_0723_0030.zip')
    dreamer_path = DATA_FOLDER + dreamer_path
    model_path = DATA_FOLDER + model_path
    record_path = DATA_FOLDER + record_path


    record_path2 = 'env_data_straight2_6x6.csv'
    record_path2 = DATA_FOLDER + record_path2



    # env.setMultiObstacles(True)
    # name = 'test'
    # main(
    #     REPLAY,
    #     models=[DATA_FOLDER + 'td3_model3_LeftTurn_stage2_0731_0859.zip'],
    #     reward_generator=RewardGeneratorLeftTurn(verbose=True),
    #     name=name,
    #     record_path=DATA_FOLDER + f'env_data_{name.lower()}_6x6.csv',
    #     record_verbose=False)
    # exit(0)

    # region TRAINING AND REPLAYING
    # train_list = [
    #     ('RightTurn', RewardGeneratorRightTurn(verbose=True)),
    #     ('LeftTurn', RewardGeneratorLeftTurn(verbose=True)),
    #     ('Steering', RewardGeneratorSteering(verbose=True)),
    # ]
    
    # # change the env to 1 obstacles
    # replay_list = []
    # for name, reward_generator in train_list:
    #     model_name = main(
    #         TRAIN,
    #         models=None,
    #         reward_generator=reward_generator,
    #         name=name,
    #         record_path=DATA_FOLDER + f'env_data_{name.lower()}_6x6.csv',
    #         record_verbose=False,
    #         train_steps=1e4)
    #     replay_list.append((model_name, name, reward_generator))
    # # exit(0)
    # #change the env to 4 obstacles
    # env.setMultiObstacles(True)

    # # replay_list = [
    # #     ('td3_model3_RightTurn_0730_1633.zip', 'RightTurn', RewardGeneratorRightTurn(verbose=True)),
    # # ]

    # replay_list_2 = []
    # for model_name, name, reward_generator in replay_list:
    #     model_name = main(
    #         TRAIN,
    #         models=[DATA_FOLDER + model_name],
    #         reward_generator=reward_generator,
    #         name=name+"_stage2",
    #         record_path=DATA_FOLDER + f'env_data_{name.lower()}_6x6_stage2.csv',
    #         record_verbose=False,
    #         train_steps=4e4)
    #     replay_list_2.append((model_name, name, reward_generator))
    # exit(0)

    env.setMultiObstacles(True)
    # replay_list_2 = [
    #     ('td3_model3_RightTurn_stage2_0731_0947.zip', 'RightTurn', RewardGeneratorRightTurn(verbose=True)),
    #     ('td3_model3_LeftTurn_stage2_0731_1009.zip', 'LeftTurn', RewardGeneratorLeftTurn(verbose=True)),
    #     ('td3_model3_Steering_stage2_0731_1031.zip', 'Steering', RewardGeneratorSteering(verbose=True)),
    # ]

    # for model, name, reward_generator in replay_list_2:
    #     model_paths = [DATA_FOLDER + model]
    #     main(
    #         REPLAY,
    #         models=model_paths,
    #         reward_generator=reward_generator,
    #         name=name,
    #         record_path=DATA_FOLDER + f'env_data_{name.lower()}_6x6.csv',
    #         record_verbose=False)
    # exit(0)
    # endregion
    
    main(
        # RECORD,

        # TRAIN,
        # models=None,
        # reward_generator=RewardGeneratorRightTurn(verbose=True),
        # reward_generator=RewardGeneratorLeftTurn(verbose=True),
        # reward_generator=RewardGeneratorStraight(verbose=True),

        # REPLAY,
        # models=model_paths,
        # reward_generator=RewardGeneratorSteering(verbose=True),

        DREAMER,
        dreamer_path=dreamer_path,
        models=model_paths,
        reward_generator=RewardGeneratorDreamer(verbose=False),

        name='test',
        record_path=record_path,
        record_verbose=False)
    
    
