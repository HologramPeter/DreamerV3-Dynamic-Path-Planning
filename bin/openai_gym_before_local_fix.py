# Copyright 1996-2024 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from controller import Supervisor

try:
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import TD3
    from stable_baselines3.common.env_checker import check_env
except ImportError as e:
    print(e)
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym==0.21 stable_baselines3"'
    )


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()

        # lidar = self.getSelf().getDevice("LDS-01")
        # self.__resolution = lidar.getHorizontalResolution()
        # self.__fov = lidar.getFov()
        # self.__maxRange = lidar.getMaxRange()

        # left_motor = self.getSelf().getDevice('left wheel motor')
        # motor_threshold = left_motor.getMaxVelocity()

        #ROBOT CONFIG
        self.resolution = 360
        self.motor_threshold = 6.67
        self.lidar_angle = 2 * np.pi / self.resolution  # angle per lidar reading

        lidar_maxRange = 3.5
        wheel_radius = 0.033
        wheel_base = 0.16
        self.collision_threshold = 0.2  # distance threshold for collision detection
        self.out_of_bounds_threshold = 1.75

        self.maxRange = lidar_maxRange
        self.target = np.array([0.0, 0.0])

        #TRAIN CONFIG
        ### To be added

        # Gym generic
        action_high = np.array(
            [
                1,
                1,
            ],
            dtype=np.float32
        )

        v_high = self.motor_threshold * wheel_radius # linear velocity
        w_high = self.motor_threshold * wheel_radius / wheel_base # angular velocity
        
        obs_high = np.array(
            [
                v_high,  
                w_high,
                10, #x coord
                10, #y coord
                np.pi, # heading difference
                self.maxRange, # closest obstacle coord
                self.maxRange
            ],
            dtype=np.float32
        )

        # obs_low = np.array(
        #     [
        #         -v_high,  
        #         -w_high,
        #         -10, #x coord
        #         -10, #y coord
        #         -np.pi # heading difference
        #     ] + [0] * self.resolution,
        #     dtype=np.float32
        # )

        # obs_high = np.array(
        #     [
        #         v_high,  
        #         w_high,
        #         10, #x coord
        #         10, #y coord
        #         np.pi # heading difference
        #     ],
        #     dtype=np.float32
        # )

        
        self.action_space = gym.spaces.Box(-action_high, action_high, (2,), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__left_motor = None
        self.__right_motor = None
        self.__lidar_sensor = None

        self.obstacle_node = None
        self.goal_node = None
        self.robot_node = None

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)
    
    def seed(self, seed=None):
        np.random.seed(seed)
        
    def render(self):
        return
    
    def close(self):
        return
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Objects
        self.obstacle_node = self.getFromDef("OBSTACLE")
        self.goal_node = self.getFromDef("GOAL")
        self.robot_node = self.getFromDef("ROBOT")

        # Motors
        self.__left_motor = self.getDevice('left wheel motor')
        self.__right_motor = self.getDevice('right wheel motor')
        self.__left_motor.setPosition(float('inf'))
        self.__right_motor.setPosition(float('inf'))
        self.__left_motor.setVelocity(0.0)
        self.__right_motor.setVelocity(0.0)

        # Sensors
        self.__lidar_sensor = self.getDevice("LDS-01")
        self.__lidar_sensor.enable(self.__timestep)

        # Internals
        super().step(self.__timestep)

        # randomize iniitial state
        square_size = 2.0
        half_size = square_size / 2.0

        # Set robot at random position on square border
        if np.random.rand() < 0.5:
            robot_x = np.random.choice([-half_size, half_size])
            robot_y = np.random.uniform(-half_size, half_size)
        else:
            robot_x = np.random.uniform(-half_size, half_size)
            robot_y = np.random.choice([-half_size, half_size])

        # Set solid at midpoint between goal (0, 0) and robot
        self.obstacle_x = robot_x / 2.0
        self.obstacle_y = robot_y / 2.0

        self.robot_node.getField('translation').setSFVec3f([robot_x, robot_y, 0.0])
        self.goal_node.getField('translation').setSFVec3f([0.0, 0.0, 0.0])
        self.obstacle_node.getField('translation').setSFVec3f([self.obstacle_x, self.obstacle_y, 0.0])

        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0.1, 0.5)  # adjust as needed
        self.obstacle_velocity_x = np.cos(angle)*speed / 1000.0
        self.obstacle_velocity_z = np.sin(angle)*speed / 1000.0

        self.step_lapsed = 0

        rotation = self.robot_node.getOrientation()

        #heading to goal
        goal_heading = np.arctan2(-robot_y, -robot_x)
        robot_heading = np.arctan2(rotation[3], rotation[4])
        heading_diff = (goal_heading - robot_heading + np.pi) % (2 * np.pi) - np.pi

        self.state = np.array([
            0,
            0,
            robot_x,
            robot_y,
            heading_diff,
            self.maxRange,
            self.maxRange
            ],
            dtype=np.float32)
        
        # self.state = np.array([
        #     0,
        #     0,
        #     robot_x,
        #     robot_y,
        #     heading_diff
        #     ])

        #Gym generic
        return self.state.astype(np.float32), {}

    def step(self, action):
        # Execute the action
        self.__left_motor.setVelocity(action[0]*self.motor_threshold)
        self.__right_motor.setVelocity(action[1]*self.motor_threshold)

        self.obstacle_x += self.obstacle_velocity_x * self.__timestep
        self.obstacle_y += self.obstacle_velocity_z * self.__timestep
        self.obstacle_node.getField('translation').setSFVec3f([
            self.obstacle_x,
            self.obstacle_y,
            0.0])

        super().step(self.__timestep)

        # Observation
        linear_velocity = self.robot_node.getVelocity()[0]
        angular_velocity = self.robot_node.getVelocity()[1]

        lidar_ranges = self.__lidar_sensor.getRangeImage()

        robot_pos = self.robot_node.getField('translation').getSFVec3f()
        rotation = self.robot_node.getOrientation()

        #calculate from min value in lidar_ranges
        ind_lidar = np.argmin(lidar_ranges)
        dist_to_obstacle = np.clip(lidar_ranges[ind_lidar], -self.maxRange, self.maxRange)
        lidar_angle = ind_lidar * self.lidar_angle
        obstacle_x = dist_to_obstacle * np.cos(lidar_angle)
        obstacle_y = dist_to_obstacle * np.sin(lidar_angle)

        dist_to_goal = np.linalg.norm(np.array([robot_pos[:2]]) - self.target)

        #heading to goal
        goal_heading = np.arctan2(-robot_pos[1], -robot_pos[0])
        robot_heading = np.arctan2(rotation[3], rotation[4])
        heading_diff = (goal_heading - robot_heading + np.pi) % (2 * np.pi) - np.pi

        self.state = np.array([
            linear_velocity,
            angular_velocity,
            robot_pos[0],
            robot_pos[1],
            heading_diff,
            obstacle_x,
            obstacle_y
            ])
        
        # self.state = np.concatenate((self.state, lidar_ranges))

        collision = dist_to_obstacle < self.collision_threshold
        reach_goal = dist_to_goal < self.collision_threshold
        out_of_bounds = dist_to_goal > self.out_of_bounds_threshold

        terminated = bool(
            collision or
            reach_goal or
            out_of_bounds
        )

        truncated = self.step_lapsed >= 2000

        self.step_lapsed += 1

        # Reward
        # collision -10
        # goal reached + 10
        # robot distance to goal +0.1
        # robot distance to solid exponential decay
        # linear velocity +0.01
        # angular velocity > 0.5 => -3.0
        # heading to goal +0.01
        # time penalty -0.01

        # Traininig seciton 1
        # linear_reward = -1 if < 0, 0 if < 0.015, 1 otherwise
        # reward = collision * 100.0 + (1.0 - collision) * (
        #     - 100.0 * out_of_bounds
        #     + 100.0 * reach_goal
        #     - 10.0 * dist_to_goal
        #     - 10.0 / ((dist_to_obstacle*10)**2 + 0.1)
        #     + 50.0 * linear_velocity * linear_reward
        #     - 100.0 * abs(angular_velocity) * (abs(angular_velocity) > 0.05)
        #     - 10.0 * abs(heading_diff)
        #     - 2e-5 * self.__timestep
        # )

        
        linear_reward = 0.0
        if linear_velocity < 0:
            linear_reward = -0.2
        elif linear_velocity < 0.02:
            linear_reward = 0.5
        else:
            linear_reward = 0.2 + linear_velocity*20


        reward = collision * 300.0 + (1.0 - collision) * (
            - 300.0 * out_of_bounds
            + 100.0 * reach_goal
            - 10.0 * dist_to_goal
            - 10.0 / ((dist_to_obstacle*10)**2 + 0.1)
            + 60.0 * linear_velocity * linear_reward
            - 100.0 * abs(angular_velocity) * (abs(angular_velocity) > 0.05)
            - 10.0 * abs(heading_diff)
            - 5e-5 * self.__timestep
        )

        

        #padding with 4 dp
        print(
              f"Reward: {reward:.4f}, "
            #   f"lidar reading: {lidar_ranges[ind_lidar]:.4f}, "
            #   f"Dist to Goal: {dist_to_goal:.4f}, "
            #   f"dist rew: {(1/((dist_to_obstacle*10)**2 + 0.1)):.4f}, "
            #   f"Dist to Obstacle: {dist_to_obstacle:.4f}, "

              f"linear_velocity: {linear_velocity:.4f}, "
              f"angular_velocity: {angular_velocity:.4f}, "
            #   f"heading_diff: {heading_diff:.4f}, "

            #   f"Heading: {robot_heading:.4f}, "
            #   f"Collision: {collision}, "
            #   f"Reach Goal: {reach_goal}, "
            #   f"Out of Bounds: {out_of_bounds}"
              )
        
        if truncated:
            self.reset()

        return self.state.astype(np.float32), reward, terminated, truncated, {}


def main():
    # Initialize the environment
    env = OpenAIGymEnvironment()
    check_env(env)

    env.seed(42)  # Set a seed for reproducibility
    
    train = True
    if train:
        # model = TD3('MlpPolicy', env, verbose=1)
        model = TD3.load("td3_model2", env=env)
        model.learn(total_timesteps=1e4)
        
        # model.save("td3_model")
    else:
        # model = TD3.load("td3_model", env=env) #without lidar
        model = TD3.load("td3_model2", env=env)

    #td3_model2 issue: slow linear speed, slow to focus on goal


    # Replay
    print('Training is finished, press `Y` to save and replay...')
    env.wait_keyboard()
    model.save("td3_model3")


    loop = False
    obs, info = env.reset()
    for _ in range(10000):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward, info)
        if terminated:
            if loop:
                obs, info = env.reset()
            else:
                break


if __name__ == '__main__':
    main()
