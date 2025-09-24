
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env

from WebotsGymEnvironment import WebotsGymEnvironment
from WebotsGymAddon import TensorPolicyWrapper, SaveObsCallback, NullCallback, generateFileName
from WebotsReward import *

import torch

from DreamerV3 import DreamerV3

from stable_baselines3.common.noise import NormalActionNoise

DATA_FOLDER = 'data_final/'
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

    run_info = [] #(steps, success, total_reward, smoothness, heading deviation)
    accum_reward = 0
    steps = 0

    run_obs = [] #reward, headings, position, velocity, angular velocity
    while runs > 0:
        action, _states = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        run_obs.append(extractObs(reward, obs))

        accum_reward += reward
        steps += 1

        if callback:
            callback.save(action.reshape((1, -1)), obs.reshape((1, -1)), reward, terminated, truncated, info)
        if terminated or truncated:
            run_info.append(summarize(run_obs, steps, info.get('success', False)))
            obs, info = env.reset()
            run_obs = []
            accum_reward = 0
            steps = 0
            runs -= 1

    return run_info

def extractObs(reward, obs):
    linear_velocity = obs[0]  # linear velocity
    angular_velocity = obs[1]  # angular velocity
    robot_pos = obs[2:4]  # robot x, y position
    heading_sin = obs[4]  # sin, cos of robot heading relative to goal
    heading_cos = obs[5]  # sin, cos of robot heading relative to goal
    return (reward, np.arctan2(heading_sin, heading_cos), robot_pos, linear_velocity, angular_velocity)

def summarize(run_info, steps, success):
    #run_info is a list of tuples (reward, heading, position, linear_velocity, angular_velocity)
    #return (steps, success, total_reward, smoothness, heading deviation)
    total_reward = 0
    smoothness = 0
    heading_deviation = 0

    #smoothness is the variance of diff positions
    #heading_deviation is the mean of heading

    positions = [info[2] for info in run_info]

    prev_pos = None
    for info in run_info:
        reward, heading, position, linear_velocity, angular_velocity = info
        total_reward += reward
        heading_deviation += np.abs(heading)

        if prev_pos is not None:
            heading_deviation += np.abs(heading - np.arctan2(prev_pos[1], prev_pos[0]))
        prev_pos = position

    mean_smoothness = calculate_jerk_smoothness(np.stack(positions))
    mean_reward = total_reward / steps
    mean_heading_deviation = heading_deviation / steps

    return (steps, 1 if success else 0, mean_reward, mean_smoothness, mean_heading_deviation)

def calculate_jerk_smoothness(positions, dt=0.032):
    # positions: (N, 2), dt: time step between positions

    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)

    jerk_magnitude_squared = np.sum(jerk**2, axis=1)
    smoothness = np.sum(jerk_magnitude_squared) * dt

    return smoothness


def runWithDreamer(env, agents, dreamer, horizon, runs,
                   callback=None):
                   
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

    dreamer.attachPolicies(policies, env.reward_func, horizon)
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

    run_obs = [] #reward, headings, position, velocity, angular velocity
    while runs > 0:
        #predict horizon observation using dreamer
        action, dream_info = dreamer.dreamPredict(action, obs, info.get('heading', 0.0))
        obs, reward, terminated, truncated, info = env.step(action)
        run_obs.append(extractObs(reward, obs))

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
            run_info.append(summarize(run_obs, steps, info.get('success', False)))
            dreamer.resetState()
            obs, info = env.reset()
            run_obs = []

            accum_reward = 0
            steps = 0
            runs -= 1
    return run_info

def main(mode, models,
         callback,
         name="test",
         dreamer_path=None,
         train_steps=1e4,
         runs = 100,
         horizon=20):

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

            run_info_list = runWithAgent(env, agent, runs=runs, callback=callback)
            for i, run_info in enumerate(run_info_list):
                steps, success, total_reward, smoothness, heading_deviation = run_info
                #log results seperated by commas
                LOG(f"{name},{i},{steps},{total_reward},{smoothness},{heading_deviation},{success}")
            FLUSH_LOG()

    elif mode == DREAMER:
        if not dreamer_path or len(models) == 0:
            raise ValueError("Model path must be provided for Dreamer mode.")

        #DREAMER_DEFINITION
        dreamer_v3 = DreamerV3(
            embed_dim=128,
            obs_dim=14,
            action_dim=2,
            deter_dim=64,
            stoch_dim=8,
            device=torch.device("cpu"))
        dreamer_v3.load(dreamer_path)
        print(f"Dreamer model loaded from {dreamer_path}")
        
        agents = []
        for agent_path in models:
            agent = TD3.load(agent_path, env=env)
            print(f"Agent model loaded from {agent_path}")
            agents.append(agent)

        print(f"Running Dreamer with {len(agents)} agents, {horizon} horizon...")
        run_info_list = runWithDreamer(env, agents, dreamer_v3, horizon=horizon,
                                        runs=runs,
                                        callback=callback)
        for i, run_info in enumerate(run_info_list):
            steps, success, total_reward, smoothness, heading_deviation = run_info
            #log results seperated by commas
            LOG(f"{name}-{horizon},{i},{steps},{total_reward},{smoothness},{heading_deviation},{success}")
        FLUSH_LOG()

if __name__ == '__main__':
    TRAIN = 0
    REPLAY = 1
    RECORD = 2
    DREAMER = 3

    record_path = 'env_data_dreamer_6x6'
    callback = NullCallback(record_path, verbose=False)


    # env.setObstacleConfig(True, 0)
    # name = 'test'
    # main(
    #     REPLAY,
    #     models=[DATA_FOLDER + 'td3_model3_LeftTurnStationary_0807_1427.zip'],
    #     reward_generator=RewardGeneratorRightTurn(verbose=True),
    #     name=name,
    #     record_path=DATA_FOLDER + f'env_data_{name.lower()}_6x6.csv',
    #     record_verbose=False)
    # exit(0)

    # change epsilon to 0.5
    epsilon=0.5

    #################TRAINING#######################
    if False:
        replay_list = []

        for speed in [2.5]:
            for i in range (1,5):
                callback.close()
                if speed == 0:
                    callback = NullCallback(record_path, verbose=False)
                if speed == 2.5:
                    callback = SaveObsCallback(record_path + f"_train_{i}.csv", verbose=False)


                env.setObstacleConfig(i, speed, epsilon=epsilon)

                train_list = [
                    (f'Sp{speed}-Obs{i}-Right', RewardGeneratorRightTurn(verbose=False)),
                    (f'Sp{speed}-Obs{i}-Left', RewardGeneratorLeftTurn(verbose=False)),
                    (f'Sp{speed}-Obs{i}-Steering', RewardGeneratorSteering(verbose=False)),
                ]

                for name, reward_generator in train_list:
                    print(f"Training {name} with {i} obstacles")
                    env.seed(SEED)
                    env.setRewardFunction(reward_generator)
                    output_name = main(
                        TRAIN,
                        models=None,
                        callback=callback,
                        name=name,
                        train_steps=5e4)
                    replay_list.append((DATA_FOLDER + output_name,i,speed))
                    callback.nextRollout()
        # exit(0)


    #################BASE PERFORMANCE#######################

    if False:
        models = replay_list
        # models = [
        #     # 'td3_model3_Sp0-Obs1-Left_0809_1326.zip',
        #     # 'td3_model3_Sp0-Obs1-Right_0809_1259.zip',
        #     # 'td3_model3_Sp0-Obs1-Steering_0809_1352.zip',
        #     'td3_model3_Sp2.5-Obs1-Right_0809_1420.zip',
        #     'td3_model3_Sp2.5-Obs1-Left_0809_1447.zip',
        #     'td3_model3_Sp2.5-Obs1-Steering_0809_1514.zip',
        # ]
        # model_paths = [(DATA_FOLDER + model_info[0], model_info[1], model_info[2]) for model_info in models]

        for model, i, speed in models:
            callback.close()
            callback = SaveObsCallback(record_path + f"_replay_{i}.csv", verbose=False)
            env.setObstacleConfig(i, speed, epsilon=epsilon)

            print(f"Replaying {model} with {i} obstacles")
            env.seed(SEED)
            env.setRewardFunction(RewardGeneratorSteering(verbose=False))
            main(
                REPLAY,
                models=[model],
                callback=callback,
                runs=100,
                name=model
            )
            callback.nextRollout()
        exit(0)

    #################Dreamer PERFORMANCE#######################
    
    if True:

        dreamers = [
            # (DATA_FOLDER + 'dreamer/dreamer_v3_obs1_0809_1350.zip', 1, 2.5,
            #  [
            #      DATA_FOLDER + 'td3_model3_Sp2.5-Obs1-Left_0809_1849.zip',
            #      DATA_FOLDER + 'td3_model3_Sp2.5-Obs1-Right_0809_1843.zip',
            #      DATA_FOLDER + 'td3_model3_Sp2.5-Obs1-Steering_0809_1856.zip',
            #  ]),
            # (DATA_FOLDER + 'dreamer/dreamer_v3_obs2_0809_1351.zip', 2, 2.5,
            #  [
            #      DATA_FOLDER + 'td3_model3_Sp2.5-Obs2-Left_0809_1909.zip',
            #      DATA_FOLDER + 'td3_model3_Sp2.5-Obs2-Right_0809_1903.zip',
            #      DATA_FOLDER + 'td3_model3_Sp2.5-Obs2-Steering_0809_1916.zip',
            #  ]),
            # (DATA_FOLDER + 'dreamer/dreamer_v3_obs3_0809_1352.zip', 3, 2.5,
            #  [
            #      DATA_FOLDER + 'td3_model3_Sp2.5-Obs3-Left_0809_1930.zip',
            #      DATA_FOLDER + 'td3_model3_Sp2.5-Obs3-Right_0809_1923.zip',
            #      DATA_FOLDER + 'td3_model3_Sp2.5-Obs3-Steering_0809_1937.zip',
            #  ]),
            ##PICK BEST ONE FOR HORIZON TESTING
            (DATA_FOLDER + 'dreamer/dreamer_v3_obs4_0809_1353.zip', 4, 2.5,
             [
                 DATA_FOLDER + 'td3_model3_Sp2.5-Obs4-Left_0809_1949.zip',
                 DATA_FOLDER + 'td3_model3_Sp2.5-Obs4-Right_0809_1943.zip',
                 DATA_FOLDER + 'td3_model3_Sp2.5-Obs4-Steering_0809_1956.zip',
             ]),
        ]

        # #models trained with speed 0
        # models_static = [
        #     'td3_model3_Sp0-Obs1-Left_0809_1326.zip',
        #     'td3_model3_Sp0-Obs1-Right_0809_1259.zip',
        #     'td3_model3_Sp0-Obs1-Steering_0809_1352.zip',
        # ]

        #models trained with speed 2.5
        # models_dynamic = [
        #     'td3_model3_Sp2.5-Obs1-Right_0809_1420.zip',
        #     'td3_model3_Sp2.5-Obs1-Left_0809_1447.zip',
        #     'td3_model3_Sp2.5-Obs1-Steering_0809_1514.zip',
        # ]

        # model_paths = [DATA_FOLDER + model for model in models_dynamic]
        # dreamer_path = DATA_FOLDER + dreamer_path
        
        for dreamer_path, obs_num, speed, model_paths in dreamers:
            callback.close()
            callback = SaveObsCallback(record_path + f"_dreamer_obs{obs_num}.csv", verbose=False)
            env.setObstacleConfig(4, speed, epsilon=1)

            for horizon in [30]:
                env.seed(10)
                env.setRewardFunction(RewardGeneratorSteering(verbose=False))
                main(
                    DREAMER,
                    dreamer_path=dreamer_path,
                    models=model_paths,
                    callback=callback,
                    runs=100,
                    horizon=horizon,
                    name=f'dreamer_obs{obs_num}',
                )
        exit(0)

    #####################TEMPLATE#####################


    model_path = 'td3_model3_straight2_0723_0933.zip'
    # dreamer_path = 'dreamer_1/dreamer_v3_model2_0723_1406.zip.pth'
    # dreamer_path = 'dreamer_v3_model_0728_0746.zip.pth' # week 9
    # dreamer_path = 'dreamer_v3_model_week9_0731_0111.zip.pth' # week 9
    dreamer_path = 'dreamer_v3_model_0807_0133.zip.pth' # Final

    #models trained with speed 0
    models_static = [
        'td3_model3_RightTurnStationary_0807_1416.zip',
        'td3_model3_LeftTurnStationary_0807_1427.zip',
        'td3_model3_SteeringStationary_0807_1438.zip'
    ]

    #models trained with speed 2.5
    models_dynamic = [
        'td3_model3_RightTurnStationary_0807_1416.zip',
        'td3_model3_LeftTurnStationary_0807_1427.zip',
        'td3_model3_SteeringStationary_0807_1438.zip'
    ]

    model_paths = [DATA_FOLDER + model for model in models]

    # model_paths.append('data/dreamer_1/td3_model3_right_turn_0720_0040.zip')
    # model_paths.append('data/dreamer_1/td3_model3_left_0723_0030.zip')
    dreamer_path = DATA_FOLDER + dreamer_path
    model_path = DATA_FOLDER + model_path
    record_path = DATA_FOLDER + record_path


    record_path2 = 'env_data_straight2_6x6.csv'
    record_path2 = DATA_FOLDER + record_path2

    main(
        # RECORD,

        # TRAIN,
        # models=None,
        # reward_generator=RewardGeneratorRightTurn(verbose=True),
        # reward_generator=RewardGeneratorLeftTurn(verbose=True),
        # reward_generator=RewardGeneratorStraight(verbose=True),

        REPLAY,
        models=model_paths,
        reward_generator=RewardGeneratorSteering(verbose=True),

        # DREAMER,
        # dreamer_path=dreamer_path,
        # models=model_paths,
        # reward_generator=RewardGeneratorDreamer(verbose=False),

        name='test',
        record_path=record_path,
        record_verbose=False)
    
    
