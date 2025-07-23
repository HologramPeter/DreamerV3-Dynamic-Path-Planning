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
    import gym
    import numpy as np
    from stable_baselines3 import TD3
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym==0.21 stable_baselines3"'
    )

#if lidar detect anything, use the RL agent to control the robot
#else, follow heading to the goal

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

        lidar_maxRange = 3.5
        wheel_radius = 0.033
        wheel_base = 0.16

        #TRAIN CONFIG
        ### To be added

        #actionspace
        action_high = np.array(
            [
                self.motor_threshold,
                self.motor_threshold,
            ],
            dtype=np.float32
        )

        v_high = self.motor_threshold * wheel_radius # linear velocity
        w_high = self.motor_threshold * wheel_radius / wheel_base # angular velocity

        
        obs_high = np.array(
            [
                v_high,  
                w_high,  
            ] + [lidar_maxRange] * self.resolution,
            dtype=np.float32
        )

        obs_low = np.array(
            [
                v_high,  
                w_high,  
            ] + [0] * self.resolution,
            dtype=np.float32
        )


        self.action_space = gym.spaces.Box(-action_high, action_high, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__left_motor = None
        self.__right_motor = None
        self.__lidar_sensor = None

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        robot = self.getSelf()

        # Motors
        self.__left_motor = robot.getDevice('left wheel motor')
        self.__right_motor = robot.getDevice('right wheel motor')
        self.__left_motor.setPosition(float('inf'))
        self.__right_motor.setPosition(float('inf'))
        self.__left_motor.setVelocity(0.0)
        self.__right_motor.setVelocity(0.0)

        # Sensors
        self.__lidar_sensor = robot.getDevice("LDS-01")
        self.__lidar_sensor.enable(self.__timestep)

        # Internals
        super().step(self.__timestep)

        square_size = 2.0
        half_size = square_size / 2.0

        # Set goal at origin
        self.goal_node.getField('translation').setSFVec3f([0.0, 0.0, 0.0])
        # Set robot at random position on square border
        if np.random.rand() < 0.5:
            robot_x = np.random.choice([-half_size, half_size])
            robot_z = np.random.uniform(-half_size, half_size)
        else:
            robot_x = np.random.uniform(-half_size, half_size)
            robot_z = np.random.choice([-half_size, half_size])
        self.robot_node.getField('translation').setSFVec3f([robot_x, 0.0, robot_z])

        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0.01, 0.05)  # adjust as needed
        self.solid_direction = [np.cos(angle), np.sin(angle)]
        self.solid_speed = speed

        # Set solid at midpoint between goal (0, 0) and robot
        solid_x = robot_x / 2.0
        solid_z = robot_z / 2.0
        self.solid_node.getField('translation').setSFVec3f([solid_x, 0.0, solid_z])


        # Open AI Gym generic
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        # action is 2-tuple: (left_wheel_speed, right_wheel_speed)
        # Execute the action
        self.__left_motor.setVelocity(action[0])
        self.__right_motor.setVelocity(action[1])
        super().step(self.__timestep)

        # Observation TODO
        robot = self.getSelf()
        endpoint = self.getFromDef("POLE_ENDPOINT")
        self.state = np.array([robot.getVelocity()[0],
                               robot.getAngularVelocity()[0], #todo verify axis
                               ]+ self.__lidar.getRangeImage())
        
        pos = self.solid_node.getField('translation').getSFVec3f()
        dx = self.solid_direction[0] * self.solid_speed
        dz = self.solid_direction[1] * self.solid_speed
        new_pos = [pos[0] + dx, pos[1], pos[2] + dz]
        self.solid_node.getField('translation').setSFVec3f(new_pos)

        robot_pos = self.robot_node.getField('translation').getSFVec3f()
        solid_pos = self.solid_node.getField('translation').getSFVec3f()

        dist = np.linalg.norm(np.array(robot_pos[:3:2]) - np.array(solid_pos[:3:2]))
        collision = dist < self.collision_threshold  # e.g., 0.1

        # Done
        done = bool(
            collision or
            #robot position at goal
            (abs(robot_pos[0]) < 0.1 and abs(robot_pos[2]) < 0.1)
        )

        # Reward
        # collision -10
        # goal reached + 10
        # robot distance to goal +0.1
        # robot distance to solid exponential decay
        # angular velocity -0.01
        # robot velocity +0.01

        reward = collision * -10.0 + \
                 (1.0 - collision) * (10.0 if done else 0.0) + \
                 (1.0 - collision) * (-abs(robot_pos[0]) - abs(robot_pos[2])) + \
                 (1.0 - collision) * np.exp(-dist) + \
                 -abs(self.state[1]) * 0.01 + \
                 abs(self.state[0]) * 0.01

        return self.state.astype(np.float32), reward, done, {}

#record info the environment as a csv file
def record_env(env, steps, repeat_action, file_path):
    with open(file_path, 'w') as f:
        f.write('rollout,sequence,observation,reward,done\n')
        env.reset()
        rollout = 0
        seq = 0

        for _ in range(steps // repeat_action):
            seq = 0

            action = env.action_space.sample()
            for _ in range(repeat_action):
                obs, reward, done, _ = env.step(action)
                f.write(f"{rollout},{seq},{obs.tolist()},{reward},{done}\n")
                seq += 1

                if done:
                    env.reset()
                    rollout += 1
                    seq = 0
                    print(f"Finished #{rollout} rollout with {seq} sequences")
        

# TODO: implement reward function
# def reward(obs):
#     collision = obs[0] < 0.1  # Example threshold for collision
#     robot_pos = obs[1:3]  # Assuming the first two elements are the
#     dist = np.linalg.norm(robot_pos[:2])  # Distance to the origin (goal)
#     done = bool(
#         collision or
#         # Robot position at goal
#         (abs(robot_pos[0]) < 0.1 and abs(robot_pos[1]) < 0.1)
#     )
#     velocity = obs[3]  # Assuming the fourth element is the robot's velocity
#     angular_velocity = obs[4]  # Assuming the fifth element is the angular velocity
#     return collision * -10.0 + \
#                  (1.0 - collision) * (10.0 if done else 0.0) + \
#                  (1.0 - collision) * (-abs(robot_pos[0]) - abs(robot_pos[2])) + \
#                  (1.0 - collision) * np.exp(-dist) + \
#                  -abs(angular_velocity) * 0.01 + \
#                  abs(velocity) * 0.01


def main():
    # Initialize the environment
    env = OpenAIGymEnvironment()
    check_env(env)

    env.reset()

    # # Train
    # model = TD3('MlpPolicy', env, n_steps=2048, verbose=1)
    # model.learn(total_timesteps=1e5)

    # # Replay
    # print('Training is finished, press `Y` for replay...')
    # env.wait_keyboard()

    # obs = env.reset()
    # for _ in range(100000):
    #     action, _states = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     print(obs, reward, done, info)
    #     if done:
    #         obs = env.reset()


if __name__ == '__main__':
    main()
