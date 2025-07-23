
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

from controller import Supervisor

import gymnasium as gym
import numpy as np

#if lidar detect anything, use the RL agent to control the robot
#RL agent resets if it cannot avoid obstacle within 20 x 20 bound
#else, follow heading to the goal

#current state deal with 1 obstacle


class WebotsGymEnvironment(Supervisor, gym.Env):
    DEFAULT_REWARD_FUNCTION = lambda obs: 0.0  # Default reward function

    def __init__(self, reward_func=DEFAULT_REWARD_FUNCTION, max_episode_steps=1000):
        super().__init__()

        #region CONFIGURATION
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
        self.sector_count = 4  # 4 sectors: front, left, rear, right

        sector_size = self.resolution // self.sector_count
        self.sector_size_half = sector_size // 2  # half sector size for bearing calculation
        self.sectors = [(i*sector_size-self.sector_size_half, i*sector_size+self.sector_size_half) for i in range(self.sector_count)]

        lidar_maxRange = 3.5
        self.wheel_radius = 0.033
        self.wheel_base = 0.16
        self.collision_threshold = 0.15  # distance threshold for collision detection
        self.bounds_threshold = 1.5

        self.maxRange = lidar_maxRange
        self.maxReading = lidar_maxRange + 1  # +1 for infinite ranges
        self.target = np.array([0.0, 0.0])

        self.reward_func = reward_func

        v_high = self.motor_threshold * self.wheel_radius # linear velocity
        w_high = 2 * self.motor_threshold * self.wheel_radius / self.wheel_base # angular velocity

        #TRAIN CONFIG
        ### To be added
        #endregion

        #region GYM SPACES
        action_high = np.array(
            [
                1, 1,
            ],
            dtype=np.float32
        )

        obs_low = np.array(
            [
                -v_high, -w_high,
                -self.bounds_threshold, -self.bounds_threshold, #x, y coord from goal
                -1, -1 #sin, cos of robot heading relative to goal
            # ] + [0] * self.resolution,
            ] + [0] * self.sector_count + [-self.sector_size_half] * self.sector_count,
            dtype=np.float32
        )

        obs_high = np.array(
            [
                v_high, w_high,
                self.bounds_threshold, self.bounds_threshold, #x, y coord from goal
                1, 1 #sin, cos of robot heading relative to goal
            # ] + [self.maxReading] * self.resolution,
             ] + [self.maxReading] * self.sector_count + [self.sector_size_half] * self.sector_count, #the no-obstacle reading
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(-action_high, action_high, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        #endregion

        #region ENV INIT
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)
        self.state = None

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
        #endregion

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

        #region RESET SIMULATION
        
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
        self.step_lapsed = 0

        # Sensors
        self.__lidar_sensor = self.getDevice("LDS-01")
        self.__lidar_sensor.enable(self.__timestep)

        # Internals
        super().step(self.__timestep)

        #endregion

        #region RANDOMIZE INITIAL STATE

        # randomize iniitial state
        square_size = 2.0
        half_size = square_size / 2.0

        # Set robot at random position on square border
        distance = 0
        while distance < 0.5 or distance > 1.0: # obj too close to robot
            if np.random.rand() < 0.5:
                robot_x = np.random.choice([-half_size, half_size])
                robot_y = np.random.uniform(-half_size, half_size)
            else:
                robot_x = np.random.uniform(-half_size, half_size)
                robot_y = np.random.choice([-half_size, half_size])

            angle = np.random.uniform(0, 2 * np.pi)
            # adjust as needed
            speed = np.random.uniform(0.05, 0.1)
            self.obstacle_velocity_x = np.cos(angle)*speed / 1000.0 * self.__timestep
            self.obstacle_velocity_y = np.sin(angle)*speed / 1000.0 * self.__timestep

            # Set solid at midpoint between goal (0, 0) and robot
            self.obstacle_x = robot_x / 2.0 + self.obstacle_velocity_x * 50
            self.obstacle_y = robot_y / 2.0 + self.obstacle_velocity_y * 50

            distance = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([self.obstacle_x, self.obstacle_y]))

        self.robot_node.getField('translation').setSFVec3f([robot_x, robot_y, 0.0])
        self.goal_node.getField('translation').setSFVec3f([0.0, 0.0, 0.0])
        self.obstacle_node.getField('translation').setSFVec3f([self.obstacle_x, self.obstacle_y, 0.0])

        #turn the robot to face the goal
        rotation = self.robot_node.getOrientation()
        robot_heading = np.arctan2(rotation[3], rotation[4])
        goal_heading = np.arctan2(-robot_y, -robot_x)

        heading_diff = (goal_heading - robot_heading + np.pi) % (2 * np.pi) - np.pi
        self.robot_node.getField('rotation').setSFRotation([0.0, 0.0, 1.0, heading_diff])

        #endregion

        #region INITIALIZE STATE

        self.state = np.array([
            0.0,  # linear velocity
            0.0,  # angular velocity
            robot_x,  # robot x position
            robot_y,  # robot y position
            np.sin(heading_diff),  # sin of heading diff
            np.cos(heading_diff),  # cos of heading diff
        # ] + [self.maxReading] * self.resolution,  # lidar readings
        ] + [self.maxReading] * self.sector_count + [self.sector_size_half] * self.sector_count,  # lidar readings
        dtype=np.float32)
        #endregion

        return self.state, {}

    def get_v_w(self, v_left, v_right):
        v = self.wheel_radius * (v_left + v_right) / 2
        w = self.wheel_radius * (v_right - v_left) / self.wheel_base
        return v, w
    


    #downsample lidar reading to front,left,right,rear
    #each sector contains minimum reading, and and bearing to that reading

    def encode_lidar_readings(self, lidar_ranges):
        """
        Encode lidar readings into a more manageable format.
        This function takes the full lidar range readings and encodes them into sectors.
        """
        encoded_readings = np.zeros(2*self.sector_count, dtype=np.float32)

        for i, (start,end) in enumerate(self.sectors):
            if i == 0:
                sector_readings = np.concatenate((lidar_ranges[start:], lidar_ranges[:end]))
            else:
                sector_readings = lidar_ranges[start:end]
            min_i = np.argmin(sector_readings)
            encoded_readings[i] = np.clip(sector_readings[min_i], 0, self.maxReading)  # minimum reading in the sector
            encoded_readings[i + self.sector_count] = min_i - self.sector_size_half  # bearing to the minimum reading in the sector
        return encoded_readings

    def step(self, action):
        # Execute the action
        left_motor_speed = action[0] * self.motor_threshold
        right_motor_speed = action[1] * self.motor_threshold
        self.__left_motor.setVelocity(left_motor_speed)
        self.__right_motor.setVelocity(right_motor_speed)

        # self.obstacle_x += self.obstacle_velocity_x
        # self.obstacle_y += self.obstacle_velocity_y
        self.obstacle_node.getField('translation').setSFVec3f([
            self.obstacle_x,
            self.obstacle_y,
            0.0])
        if self.step_lapsed % 400 == 0:
            self.obstacle_velocity_x *= -1
            self.obstacle_velocity_y *= -1


        super().step(self.__timestep)

        # Observation
        linear_velocity, angular_velocity = self.get_v_w(left_motor_speed, right_motor_speed)
        lidar_ranges = self.__lidar_sensor.getRangeImage()
        lidar_ranges = self.encode_lidar_readings(lidar_ranges)

        robot_pos = self.robot_node.getField('translation').getSFVec3f()[:2] - self.target

        rotation = self.robot_node.getOrientation()
        robot_heading = np.arctan2(rotation[3], rotation[4])
        goal_heading = np.arctan2(-robot_pos[1], -robot_pos[0])

        heading_diff = (goal_heading - robot_heading + np.pi) % (2 * np.pi) - np.pi

        self.state = np.array([
            linear_velocity,
            angular_velocity,
            robot_pos[0],
            robot_pos[1],
            np.sin(heading_diff),
            np.cos(heading_diff),
        ], dtype=np.float32)
        self.state = np.concatenate((self.state, lidar_ranges))
        
        # Termination
        dist_to_obstacle = np.linalg.norm(np.array([robot_pos[0] - self.obstacle_x, robot_pos[1] - self.obstacle_y]))
        dist_to_goal = np.linalg.norm(robot_pos)

        # check out of bound from pos x, y within 10 and -10
        out_of_bounds = (
            abs(robot_pos[0]) > self.bounds_threshold or
            abs(robot_pos[1]) > self.bounds_threshold
        )
        
        terminated = bool(
            dist_to_obstacle < self.collision_threshold or  # collision with obstacle
            out_of_bounds or  # out of bounds
            dist_to_goal < self.collision_threshold  # reached goal
        )

        # Truncated
        truncated = self.step_lapsed >= self.spec.max_episode_steps
        self.step_lapsed += 1

        # Reward
        reward = self.reward_func(self.state)
        if terminated:
            reward += 500.0 if dist_to_goal < self.collision_threshold else -500.0
        reward -= 1e-2 * (self.step_lapsed // 10)  # time penalty

        return self.state.astype(np.float32), reward, terminated, truncated, {}