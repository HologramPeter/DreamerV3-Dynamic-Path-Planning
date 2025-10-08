
import numpy as np
import pandas as pd
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env

from WebotsObstacles import ObstacleConfig
from WebotsGymEnvironment import EnvConfig, WebotsGymEnvironment
from WebotsGymAddon import TensorPolicyWrapper, SaveObsCallback, NullCallback, generateFileName
from WebotsReward import *

import torch

from DreamerV3 import DreamerV3

from stable_baselines3.common.noise import NormalActionNoise



####################### SETTINGS ########################
policy_kwargs = dict(
    net_arch=[256, 256],
    # activation_fn=torch.nn.ReLU(),
)

################################################
    
####################### LOGGING ########################
DATA_FOLDER = 'data2/'
getPath = lambda filename: DATA_FOLDER + filename

RECORD_PATH = getPath('env_data_dreamer_6x6.log')
callback = NullCallback(RECORD_PATH, verbose=False)
def TOGGLE_CALLBACK(isOn):
    global callback
    callback.close()
    if isOn:
        callback = SaveObsCallback(RECORD_PATH, verbose=False)
    else:
        callback = NullCallback(RECORD_PATH, verbose=False)

DATA_LOG_PATH = getPath('data.log')
columns = ["name", "run_num", "steps", "total_reward", "smoothness", "heading_deviation", "success"]
df = pd.DataFrame([], columns=columns)
df.to_csv(DATA_LOG_PATH, index=False)

def LOG_DATA(df):
    df.to_csv(DATA_LOG_PATH, mode='a', header=False, index=False)


LOG_FILE_PATH = getPath('log.log')
LOG_FILE = open(LOG_FILE_PATH, 'a')
LOG = lambda *arg: print(*arg, file=LOG_FILE)
def FLUSH_LOG():
    LOG_FILE.flush()

LOG_FILE = open(LOG_FILE_PATH, 'a')


################################################

####################### SEED ########################
# 982040
# Rand = np.random.randint(0, 1000000)
# SEED = Rand
# SEED = 982040
# print(f"Random seed: {SEED}")
SEED = 42
################################################


####################### ENV ########################
area = 3.0  # 6x6 area

obstacleConfig = ObstacleConfig(
    # x_range=(-0, 0),
    # y_range=(-0, 1),
    x_range=(-area, area),
    y_range=(-area, area),
    x_speed_range=(-2.5, -2.5),
    y_speed_range=(-2.5, -2.5),
    period_range=(50, 200),
    count=10,
)

envConfig = EnvConfig(
    bounds_threshold=3.0,
    collision_threshold=0.2,
    obstacle_config=obstacleConfig,
    max_episode_steps=2000,
    reward_func=None
)

env = WebotsGymEnvironment(envConfig)
# check_env(env)
env.seed(SEED)

################################################

####################### Utility ########################
def extractObs(reward, obs):
    linear_velocity = obs[0]  # linear velocity
    angular_velocity = obs[1]  # angular velocity
    robot_pos = obs[2:4]  # robot x, y position
    heading_sin = obs[4]  # sin, cos of robot heading relative to goal
    heading_cos = obs[5]  # sin, cos of robot heading relative to goal
    return (reward, np.arctan2(heading_sin, heading_cos), robot_pos, linear_velocity, angular_velocity)

def summarize(name, run_count, success, steps, run_info):
    run_info = list(map(lambda x: extractObs(x[1], x[0]), run_info))

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

    return pd.DataFrame([
        name, run_count, steps, 1 if success else 0,
        mean_reward, mean_smoothness, mean_heading_deviation], columns=columns)

def calculate_jerk_smoothness(positions, dt=0.032):
    # positions: (N, 2), dt: time step between positions

    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)

    jerk_magnitude_squared = np.sum(jerk**2, axis=1)
    smoothness = np.sum(jerk_magnitude_squared) * dt

    return smoothness

################################################



#record info the environment
def runRandomEnv(steps, repeat_steps):
        obs = env.reset()
        for _ in range(steps // repeat_steps):
            action = env.action_space.sample()
            for _ in range(repeat_steps):
                obs, reward, terminated, truncated, info = env.step(action)
                callback.save(np.array(action), obs, reward, terminated, truncated, info)
                if terminated or truncated:
                    env.reset()

def replay(model, name, runs):
    agent = TD3.load(getPath(model), env=env)
    print(f"{model} loaded for replay")
    
    obs, info = env.reset()
    accum_reward = 0
    steps = 0
    run_count = 0

    run_obs = []  #(obs, reward)
    while run_count < runs:
        action, _states = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        run_obs.append((reward, obs))
        accum_reward += reward
        steps += 1

        if callback:
            callback.save(action.reshape((1, -1)), obs.reshape((1, -1)), reward, terminated, truncated, info)
        if terminated or truncated:
            LOG_DATA(summarize(name, run_count, info.get('success', False), steps, run_obs))
            obs, info = env.reset()
            run_obs = []
            accum_reward = 0
            steps = 0
            run_count += 1

def runWithDreamer(name, dreamer, models, horizon, runs):
    dreamer_v3 = DreamerV3(
            embed_dim=128,
            obs_dim=14,
            action_dim=2,
            deter_dim=64,
            stoch_dim=8,
            device=torch.device("cpu"))
    dreamer_v3.load(getPath(dreamer))
    print(f"Dreamer model loaded from {dreamer}")
        
    agents = []
    for model in models:
        agent = TD3.load(getPath(model), env=env)
        print(f"Agent model loaded from {model}")
        agents.append(agent)

    print(f"Running Dreamer with {len(agents)} agents, {horizon} horizon...")

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
    accum_reward = 0
    steps = 0
    run_count = 0

    getLine = lambda policy_info: [(_obs[2]*env.bounds_threshold, _obs[3]*env.bounds_threshold, 0.1) for _obs in policy_info.get('imagined_obs_seq', [])]

    run_obs = [] #reward, headings, position, velocity, angular velocity
    while run_count < runs:
        #predict horizon observation using dreamer
        action, dream_info = dreamer.dreamPredict(action, obs, info.get('heading', 0.0))
        obs, reward, terminated, truncated, info = env.step(action)
        run_obs.append((reward, obs))
        accum_reward += reward
        steps += 1

        if len(dream_info) > 0:
            lines = []
            for policy_index, policy_info in dream_info.items():
                lines.append((policy_index == dreamer.current_policy_index, getLine(policy_info)))
            env.drawLines(lines)

        if callback:
            callback.save(action.reshape((1, -1)), obs.reshape((1, -1)), reward, terminated, truncated, info)
        if terminated or truncated:
            LOG_DATA(summarize(name, run_count, info.get('success', False), steps, run_obs))
            obs, info = env.reset()
            dreamer.resetState()
            run_obs = []
            accum_reward = 0
            steps = 0
            run_count += 1

def train(model, name, train_steps):
    if model:
        agent = TD3.load(model, env=env)
        print(f"Model loaded from {model}")
    else:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

        agent = TD3('MlpPolicy', env, policy_kwargs=policy_kwargs,
                    train_freq=1, gradient_steps=1,
                    action_noise=action_noise
                )
        print("New agent created")

    agent.learn(total_timesteps=train_steps, callback=callback)

    name = generateFileName(prefix='td3_' + name)
    agent.save(getPath(name))
    print(f"Model saved as {name}")
    return name

if __name__ == '__main__':
    #################TRAINING#######################

    def train_section(steps, verbose):
        replay_list = []

        for speed in [2.5]:
            # for i in [0]:
            for i in range (10,11):
                TOGGLE_CALLBACK(speed == 2.5)
                obstacleConfig.count = i
                obstacleConfig.x_speed_range = (-speed, speed)
                obstacleConfig.y_speed_range = (-speed, speed)
                env.setObstacleConfig(obstacleConfig)

                train_list = [
                    (f'Sp{speed}-Obs{i}-Right', RewardGeneratorRightTurn(verbose)),
                    # (f'Sp{speed}-Obs{i}-Left', RewardGeneratorLeftTurn(verbose)),
                    # (f'Sp{speed}-Obs{i}-Steering', RewardGeneratorSteering(verbose)),
                ]

                for name, reward_generator in train_list:
                    print(f"Training {name}")
                    env.setRewardFunction(reward_generator)
                    env.seed(SEED)
                    output_path = train(
                        model=None,
                        name=name,
                        train_steps=steps)
                    replay_list.append((i, speed, reward_generator,output_path))
                    callback.nextRollout()
        return replay_list
    
    replay_list = train_section(
        steps=5e4,
        verbose=True
        )
    exit(0)

    #################BASE PERFORMANCE#######################

    def base_section(model_list):
        for i, speed, reward_generator, model in model_list:
            obstacleConfig.count = i
            obstacleConfig.x_speed_range = (-speed, speed)
            obstacleConfig.y_speed_range = (-speed, speed)
            env.setObstacleConfig(obstacleConfig)

            print(f"Replaying {model} with {i} obstacles")
            env.setRewardFunction(reward_generator)
            env.seed(SEED)
            replay(
                model=model,
                name=model,
                runs=100
            )
            callback.nextRollout()
    
    models = replay_list
    # MANUAL SELECTION
    # models = [
    #     (1, 2.5, RewardGeneratorSteering(), 'td3_Sp0-Obs1-Left_0809_1326.zip'),
    #     (1, 2.5, RewardGeneratorSteering(), 'td3_Sp0-Obs1-Right_0809_1259.zip'),
    #     (1, 2.5, RewardGeneratorSteering(), 'td3_Sp0-Obs1-Steering_0809_1352.zip'),
    #     (1, 2.5, RewardGeneratorSteering(), 'td3_Sp2.5-Obs1-Right_0809_1420.zip'),
    #     (1, 2.5, RewardGeneratorSteering(), 'td3_Sp2.5-Obs1-Left_0809_1447.zip'),
    #     (1, 2.5, RewardGeneratorSteering(), 'td3_Sp2.5-Obs1-Steering_0809_1514.zip'),
    # ]

    verbose = False
    for item in models:
        item[2].setVerbose(verbose)

    base_section(
        replay_list,
        runs=100
    )
    # exit(0)

    #################Dreamer PERFORMANCE#######################
    
    def dreamer_section(dreamers):
        for obs_num, speed, dreamer, models in dreamers:
            obstacleConfig.count = obs_num
            obstacleConfig.x_speed_range = (-speed, speed)
            obstacleConfig.y_speed_range = (-speed, speed)
            env.setObstacleConfig(obstacleConfig)

            reward_generator = RewardGeneratorSteering(verbose=False)

            for horizon in [30]:
                env.setRewardFunction(reward_generator)
                env.seed(SEED)
                runWithDreamer(
                    name=f'dreamer_obs{obs_num}',
                    dreamer_path=dreamer,
                    models=models,
                    horizon=horizon,
                    runs=100,
                )
                callback.nextRollout()

    dreamers = ( 1, 2.5, 'dreamer/dreamer_v3_obs1_0809_1350.zip',
                list(map(lambda x: x[-1], replay_list)))

    dreamers = [
            ( 1, 2.5, 'dreamer/dreamer_v3_obs1_0809_1350.zip', 
             [
                'td3_Sp2.5-Obs1-Left_0809_1849.zip',
                'td3_Sp2.5-Obs1-Right_0809_1843.zip',
                'td3_Sp2.5-Obs1-Steering_0809_1856.zip',
             ]),
        ]

    dreamer_section(dreamers)
    # exit(0)

    ##########################################