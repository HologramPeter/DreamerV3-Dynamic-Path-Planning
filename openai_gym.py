
import os
import torch

import numpy as np
import pandas as pd
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise

from WebotsPlanner import PlannerConfig
from WebotsObstacles import ObstacleConfig
from WebotsGymEnvironment import EnvConfig, WebotsGymEnvironment
from WebotsGymAddon import TensorPolicyWrapper, SaveObsCallback, NullCallback, generateFileName
from WebotsReward import *

from DreamerV3 import DreamerV3
    
####################### LOGGING ########################
DATA_FOLDER = 'data_9992_1/'
MODEL_FOLDER = 'models/'
MODEL_PATH = DATA_FOLDER + MODEL_FOLDER
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

getPath = lambda filename: DATA_FOLDER + filename
getModelPath = lambda filename: MODEL_PATH + filename

def getModels():
    replay_list = []
    for f in os.listdir(MODEL_PATH):
        if f.endswith('.zip'):
            replay_list.append(f)
            print(f)
    return replay_list

RECORD_PATH = getPath('env_data_dreamer_6x6.log')
callback = NullCallback(RECORD_PATH, verbose=False)
def TOGGLE_CALLBACK(isOn):
    global callback
    callback.close()
    if isOn:
        callback = SaveObsCallback(RECORD_PATH, verbose=False)
    else:
        callback = NullCallback(RECORD_PATH, verbose=False)

DATA_LOG_PATH = getPath('performance.log')
columns = ["name", "run_num", "steps", "success", "total_reward", "smoothness", "heading_deviation"]
if not os.path.exists(DATA_LOG_PATH):
    df = pd.DataFrame([], columns=columns)
    df.to_csv(DATA_LOG_PATH, index=False)

def LOG_DATA(df):
    df.to_csv(DATA_LOG_PATH, mode='a', header=False, index=False)
    print("############################ RUN COMPLETE ############################")


LOG_FILE_PATH = getPath('log.log')
LOG_FILE = open(LOG_FILE_PATH, 'a')
LOG = lambda *arg: print(*arg, file=LOG_FILE)
def FLUSH_LOG():
    LOG_FILE.flush()

LOG_FILE = open(LOG_FILE_PATH, 'a')


################################################

####################### SETTINGS ########################
policy_kwargs = dict(
    net_arch=[256, 256],
    # activation_fn=torch.nn.ReLU(),
)

STEPS = 5e4
RUNS = 100

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
halfSize = 3.0  # 6x6 area
# 1m beyond the planner area
runArea = 10.0

trainObstacleMult = 1.0
testObstacleMult = 2.25 #8x8 turn into 12 by 12

trainObstacleConfig = ObstacleConfig(
    x_range=(-halfSize-1, halfSize+1),
    y_range=(-halfSize-1, halfSize+1),
    x_speed_range=(-2.5, -2.5),
    y_speed_range=(-2.5, -2.5),
    period_range=(50, 200),
    count=10,
    collision_threshold=0.2
)

runObstacleConfig = ObstacleConfig(
    x_range=(-halfSize-1, runArea+1),
    y_range=(-halfSize-1, runArea+1),
    x_speed_range=(-2.5, -2.5),
    y_speed_range=(-2.5, -2.5),
    period_range=(50, 200),
    count=10,
    collision_threshold=0.2
)

trainPlannerConfig = PlannerConfig(
    start=np.array([0.0, 0.0]),
    target=np.array([halfSize, 0.0]),
    half_size=halfSize
)

runPlannerConfig = PlannerConfig(
    start=np.array([0.0, 0.0]),
    target=np.array([runArea, runArea]),
    half_size=halfSize
)

envConfig = EnvConfig(
    obstacle_config=trainObstacleConfig,
    planner_config=trainPlannerConfig,
    max_episode_steps=2000,
    reward_func=None
)

def setEnvTrainingEnabled(enabled: bool):
    env.setPlannerConfig(trainPlannerConfig if enabled else runPlannerConfig)
    env.setTruncationEnabled(enabled)
    return (trainObstacleConfig, trainObstacleMult) if enabled else (runObstacleConfig, testObstacleMult)

env = WebotsGymEnvironment(envConfig)
# check_env(env)
env.seed(SEED)

################################################

####################### Utility ########################
def extractObs(obs, reward):
    linear_velocity = obs[0]  # linear velocity
    angular_velocity = obs[1]  # angular velocity
    robot_pos = obs[2:4]  # robot x, y position
    heading_sin = obs[4]  # sin, cos of robot heading relative to goal
    heading_cos = obs[5]  # sin, cos of robot heading relative to goal
    return (reward, np.arctan2(heading_sin, heading_cos), robot_pos, linear_velocity, angular_velocity)

def summarize(name, run_count, success, steps, run_info):
    run_info = list(map(lambda x: extractObs(x[0], x[1]), run_info))

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

    return pd.DataFrame([[
        name, run_count, steps, 1 if success else 0,
        mean_reward, mean_smoothness, mean_heading_deviation]], columns=columns)

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
    agent = TD3.load(getModelPath(model), env=env)
    print(f"{model} loaded for replay")
    
    obs, info = env.reset()
    accum_reward = 0
    steps = 0
    run_count = 0

    run_obs = []  #(obs, reward)
    while run_count < runs:
        action, _states = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        run_obs.append((obs, reward))
        accum_reward += reward
        steps += 1

        if callback:
            callback.save(action.reshape((1, -1)), obs.reshape((1, -1)), reward, terminated, truncated, info)
        if terminated or truncated:
            summary = summarize(name, run_count, info.get('success', False), steps, run_obs)
            LOG_DATA(summary)
            obs, info = env.reset()
            run_obs = []
            accum_reward = 0
            steps = 0
            run_count += 1
            #print summary with column names
            print(summary.to_string(index=False))

def runWithDreamer(name, dreamer, models, horizon, runs, drawLines):
    # 'dreamer_v3_44_2_128_64_16_1026_0909.zip'
    # obs_dim, action_dim, embed_dim, deter_dim, stoch_dim
    dreamer_info = dreamer.split('_')
    obs_dim = int(dreamer_info[2])
    action_dim = int(dreamer_info[3])
    embed_dim = int(dreamer_info[4])
    deter_dim = int(dreamer_info[5])
    stoch_dim = int(dreamer_info[6])

    dreamer_v3 = DreamerV3(
            obs_dim=obs_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            device=torch.device("cpu"))
    dreamer_v3.load(getPath(dreamer))
    print(f"Dreamer model loaded from {getPath(dreamer)}")
        
    agents = []
    for model in models:
        agent = TD3.load(getModelPath(model), env=env)
        print(f"Agent model loaded from {getModelPath(model)}")
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

    dreamer_v3.attachPolicies(policies, env.reward_func, horizon)
    dreamer_v3.setInferredSettings(
        step_size=0.032,
        wheel_base=0.16,
        wheel_radius=0.033,
        max_wheel_speed=6.67,
    )

    dreamer_v3.resetState()
    obs, info = env.reset()

    action = np.zeros(env.action_space.shape)
    accum_reward = 0
    steps = 0
    run_count = 0

    def getLine(policy_info):
        position = []
        for _obs in policy_info.get('imagined_obs_seq', []):
            sub_obs = _obs[2:4]
            x, y = env.obs_to_global(sub_obs)
            position.append((x, y, 0.1))
        return position
    
    run_obs = []  #(obs, reward)
    while run_count < runs:
        #predict horizon observation using dreamer
        action, dream_info = dreamer_v3.dreamPredict(action, obs, info.get('heading', 0.0))
        obs, reward, terminated, truncated, info = env.step(action)

        run_obs.append((obs, reward))
        accum_reward += reward
        steps += 1

        if drawLines and len(dream_info) > 0:
            lines = []
            for policy_index, policy_info in dream_info.items():
                lines.append((policy_index == dreamer_v3.current_policy_index, getLine(policy_info)))
            env.drawLines(lines)

        if callback:
            callback.save(action.reshape((1, -1)), obs.reshape((1, -1)), reward, terminated, truncated, info)
        if terminated or truncated:
            summary = summarize(name, run_count, info.get('success', False), steps, run_obs)
            LOG_DATA(summary)
            obs, info = env.reset()
            run_obs = []
            accum_reward = 0
            steps = 0
            run_count += 1
            #print summary with column names
            print(summary.to_string(index=False))

def train(model, name, train_steps):
    if model:
        agent = TD3.load(getModelPath(model), env=env)
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

    name = generateFileName(prefix='td3-' + name)
    agent.save(getModelPath(name))
    print(f"Model saved as {name}")
    return name

if __name__ == '__main__':
    #################TRAINING#######################


    def train_section(obstacleConfig, steps, verbose):
        replay_list = {}

        for speed in [2.5]:
            # for i in [0]:
            for i in [10, 20, 30]:
                replay_list[i] = []
                obstacleConfig.count = i
                obstacleConfig.x_speed_range = (-speed, speed)
                obstacleConfig.y_speed_range = (-speed, speed)
                env.setObstacleConfig(obstacleConfig)

                train_list = [
                    (f'Sp{speed}-Obs{i}-Right', RewardGeneratorRightTurn(verbose)),
                    (f'Sp{speed}-Obs{i}-Left', RewardGeneratorLeftTurn(verbose)),
                    (f'Sp{speed}-Obs{i}-Steering', RewardGeneratorSteering(verbose)),
                ]

                for name, reward_generator in train_list:
                    print(f"Training {name}")
                    env.setRewardFunction(reward_generator)
                    env.seed(SEED)
                    output_path = train(
                        model=None,
                        name=name,
                        train_steps=steps)
                    replay_list[i].append(output_path)
                    callback.nextRollout()
        return replay_list


    # TOGGLE_CALLBACK(True)
    # obstacleConfig, obstacleMult = setEnvTrainingEnabled(True)
    # trained_model_list = train_section(
    #     obstacleConfig=obstacleConfig,
    #     steps=STEPS,
    #     verbose=True
    #     )
    # TOGGLE_CALLBACK(False)
    # env.quit()
    # exit(0)

    #################BASE PERFORMANCE#######################

    def base_section(obstacleConfig, model_list, runs):
        for i, speed, reward_generator, model in model_list:
            print("========================================")
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
                runs=runs
            )
            callback.nextRollout()

    def extractInfo(model_name, obstacleMult, verbose=False):
        parts = model_name.split('-')
        speed = float(parts[1][2:])
        obs = int(parts[2][3:])
        reward_type = parts[3]

        # reward_generator = RewardGeneratorLeftTurn(verbose)

        if reward_type == 'Left':
            reward_generator = RewardGeneratorLeftTurn(verbose)
        elif reward_type == 'Right':
            reward_generator = RewardGeneratorRightTurn(verbose)
        else:
            reward_generator = RewardGeneratorSteering(verbose)

        #multiply obs by obstacleMult and round to nearest int
        obs = int(round(obs * obstacleMult))

        return (obs, speed, reward_generator, model_name)
    



    # MANUAL SELECTION
    # replay_list = [
    #     'td3-Sp2.5-Obs30-Left-1010_1831.zip',
    #     'td3-Sp2.5-Obs30-Right-1010_1804.zip',
    #     'td3-Sp2.5-Obs30-Steering-1010_1858.zip',
    # ]

    replay_list = getModels()

    obstacleConfig, obstacleMult = setEnvTrainingEnabled(True)
    verbose = False
    models = [extractInfo(model, obstacleMult, verbose) for model in replay_list]

    # base_section(
    #     obstacleConfig=obstacleConfig,
    #     model_list=models,
    #     runs=RUNS
    # )
    # env.quit()
    # exit(0)

    #################Dreamer PERFORMANCE#######################

    def dreamer_section(obstacleConfig, obstacleMult, dreamers, runs, verbose):
        for obs_num, speed, dreamer, models in dreamers:
            obstacleConfig.count = int(obs_num * obstacleMult)
            obstacleConfig.x_speed_range = (-speed, speed)
            obstacleConfig.y_speed_range = (-speed, speed)
            env.setObstacleConfig(obstacleConfig)

            reward_generator = RewardGeneratorSteering(verbose=verbose)

            for horizon in [30]:
                env.setRewardFunction(reward_generator)
                env.seed(SEED)
                runWithDreamer(
                    name=f'dreamer_obs{obs_num}',
                    dreamer=dreamer,
                    models=models,
                    horizon=horizon,
                    runs=runs,
                    drawLines=False
                )
                callback.nextRollout()

    # dreamers = [(obstacleCount, 2.5, 'dreamer_v3_44_2_128_64_16_1026_0909.zip', models) for obstacleCount, models in trained_model_list]


    dreamers = [
            ( 10, 2.5, 'dreamer_v3_44_2_128_64_16_1026_0909.zip',
             [
                'td3-Sp2.5-Obs10-Left-1026_2341.zip',
                'td3-Sp2.5-Obs10-Right-1026_2336.zip',
                'td3-Sp2.5-Obs10-Steering-1026_2347.zip',
             ]),
             ( 20, 2.5, 'dreamer_v3_44_2_128_64_16_1026_0909.zip', 
             [
                'td3-Sp2.5-Obs20-Left-1026_2358.zip',
                'td3-Sp2.5-Obs20-Right-1026_2352.zip',
                'td3-Sp2.5-Obs20-Steering-1027_0004.zip',
             ]),
             ( 30, 2.5, 'dreamer_v3_44_2_128_64_16_1026_0909.zip', 
             [
                'td3-Sp2.5-Obs30-Left-1027_0015.zip',
                'td3-Sp2.5-Obs30-Right-1027_0009.zip',
                'td3-Sp2.5-Obs30-Steering-1027_0023.zip',
             ]),
        ]

    verbose = False
    obstacleConfig, obstacleMult = setEnvTrainingEnabled(True)
    dreamer_section(obstacleConfig, obstacleMult, dreamers, RUNS, verbose=verbose)
    env.quit()
    exit(0)

    ##########################################