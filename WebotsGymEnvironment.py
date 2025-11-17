
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
from WebotsObstacles import *
from WebotsLines import *
from WebotsPlanner import *


#TODO: add continous action space option
getPosition = lambda node: np.array(node.getField('translation').getSFVec3f()[:2])
setPosition = lambda node, pos: node.getField('translation').setSFVec3f([pos[0], pos[1], 0.0])
getRotation = lambda node: extractHeadingAngle(node.getField('rotation').getSFRotation())
setRotation = lambda node, heading: node.getField('rotation').setSFRotation([0.0, 0.0, 1.0, heading])

def extractHeadingAngle(sfRotation):
    ax, ay, az, angle = sfRotation
    # For Z-axis rotation, heading is angle * axis_z
    heading = angle * az
    # Normalize to [-pi, pi]
    heading = (heading + np.pi) % (2 * np.pi) - np.pi
    return heading

class EnvConfig:
    def __init__(self,
                 planner_config: PlannerConfig,
                 obstacle_config: ObstacleConfig,
                 max_episode_steps=1000,
                 reward_func=None):
        self.max_episode_steps = max_episode_steps
        self.planner_config = planner_config
        self.obstacle_config = obstacle_config

        if reward_func is not None:
            self.reward_func = reward_func 
        else:
            self.reward_func = lambda obs: 0.0

class WebotsGymEnvironment(Supervisor, gym.Env):
    def __init__(self, config: EnvConfig):
        super().__init__()

        #region ROBOT CONFIG
        self.resolution = 360
        self.motor_threshold = 6.67
        # self.lidar_angle = 2 * np.pi / self.resolution  # angle per lidar reading
        self.lidar_angle = 1 / self.resolution  # angle per lidar reading
        self.sector_count = 18  # number of sectors to divide the lidar readings into

        sector_size = self.resolution // self.sector_count
        self.sector_size_half = sector_size // 2  # half sector size for bearing calculation
        # self.sectors = [(i*sector_size-self.sector_size_half, i*sector_size+self.sector_size_half) for i in range(self.sector_count)]

        lidar_maxRange = 3.5
        self.wheel_radius = 0.033
        self.wheel_base = 0.16
        self.maxRange = lidar_maxRange

        self.bounds_threshold = config.planner_config.half_size  # x,y bounds for the robot
        self.collision_threshold = config.obstacle_config.collision_threshold  # distance threshold for collision detection
        self.reward_func = config.reward_func

        self.v_high = self.motor_threshold * self.wheel_radius # linear velocity
        self.w_high = 2 * self.motor_threshold * self.wheel_radius / self.wheel_base # angular velocity

        #region GYM SPACES
        action_high = np.array(
            [
                1, 1,
            ],
            dtype=np.float32
        )

        obs_low = np.array(
            [
                -1, -1, #linear, angular velocity 
                -1, -1, #x, y coord
                -1, -1, #sin, cos of robot heading
                -1, -1  #sin, cos of goal heading
                #delta lidar readings + lidar readings
            ] + [-1] * (self.sector_count)  + [0] * (self.sector_count),
            dtype=np.float32
        )

        obs_high = np.array(
            [
                1, 1, #linear, angular velocity 
                1, 1, #x, y coord
                1, 1, #sin, cos of robot heading
                1, 1  #sin, cos of goal heading
                #delta lidar readings + lidar readings
            ] + [1] * (self.sector_count)  + [1] * (self.sector_count),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(-action_high, action_high, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        #endregion

        #region ENV INIT
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=config.max_episode_steps)
        self.obs = None

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__left_motor = None
        self.__right_motor = None
        self.__lidar_sensor = None

        self.goal_node = None
        self.robot_node = None

        print(f"Time step: {self.__timestep} ms")

        self.planner = Planner(config.planner_config, self.rootChildren())
        self.lineManager = LineManager(self.rootChildren())
        self.obstacleManager = ObstacleManager(self.rootChildren(), config=config.obstacle_config)

        self.collision_threshold = config.obstacle_config.collision_threshold
        self.truncation = True
        self.step_lapsed = 0
        
        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)
        #endregion

    def seed(self, seed=None):
        np.random.seed(seed)

    def rootChildren(self):
        return self.getRoot().getField('children')
    
    def setRewardFunction(self, reward_func):
        self.reward_func = reward_func

    def setObstacleConfig(self, obstacle_config):
        self.obstacleManager.setConfig(obstacle_config)

    def setPlannerConfig(self, planner_config: PlannerConfig):
        self.planner.setConfig(planner_config)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)
        
    def render(self):
        return
    
    def close(self):
        return
    
    def quit(self):
        self.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
    
    def setTruncationEnabled(self, enabled: bool):
        self.truncation = enabled

    def obs_to_global(self, obs_position):
        return self.planner.localToGlobalPosition(obs_position * self.bounds_threshold)
     
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #region Reset the simulation
        # Objects
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

        # Physics
        self.simulationResetPhysics()
        super().step(self.__timestep)

        #endregion

        #region INITIAL STATE
        
        #NOTE: only called once per run
        self.lineManager.reset()
        self.obstacleManager.reset()
        self.planner.reset()

        #settings are in global coordinates
        #obs are in local coordinates

        target = self.planner.getCurrentTarget()
        # print("New target:", target) #TEST
        setPosition(self.goal_node, target)

        init_position, init_heading = self.planner.localToGlobal(
            position=np.random.uniform(-0.1, 0.1, size=2),
            heading=np.random.uniform(-np.pi, np.pi)
        )
        setPosition(self.robot_node, init_position)
        setRotation(self.robot_node, init_heading)

        goal_heading = np.arctan2(target[1] - init_position[1], target[0] - init_position[0]) - init_heading

        local_position, local_heading = self.planner.globalToLocal(
            position=init_position,
            heading=init_heading
        )

        self.obs = np.array([
            0.0, 0.0, #linear, angular velocity
            local_position[0], local_position[1], #x, y coord
            np.sin(local_heading), np.cos(local_heading), #sin, cos of robot heading
            np.sin(goal_heading), np.cos(goal_heading)  #sin, cos of goal heading
        ] + [0] * (self.sector_count) + [1] * (self.sector_count),  # delta lidar readings + lidar readings
        dtype=np.float32)
        #endregion

        return self.obs, {}

    def get_v_w(self, v_left, v_right):
        v = self.wheel_radius * (v_left + v_right) / 2
        w = self.wheel_radius * (v_right - v_left) / self.wheel_base
        return v, w

    #convert to range [0,1] by dividing by maxRange
    #return spacial coordinates of the min reading in each sector
    def encode_lidar_readings(self, lidar_ranges):
        """
        Encode lidar readings into a more manageable format.
        This function takes the full lidar range readings and encodes them into sectors.
        """
        #downsample lidar reading to sector_count by using circular min pooling
        encoded_readings = np.array(lidar_ranges).reshape(self.sector_count, -1)
        encoded_readings = encoded_readings.min(axis=1)
        encoded_readings = np.clip(encoded_readings / self.maxRange, 0, 1)
        
        return encoded_readings

    def step(self, action):
        # Execute the action
        left_motor_speed = action[0] * self.motor_threshold
        right_motor_speed = action[1] * self.motor_threshold
        self.__left_motor.setVelocity(left_motor_speed)
        self.__right_motor.setVelocity(right_motor_speed)
        
        self.obstacleManager.step()
        super().step(self.__timestep)

        # Observation
        last_lidar_range = self.obs[-self.sector_count:]
        linear_velocity, angular_velocity = self.get_v_w(left_motor_speed, right_motor_speed)
        lidar_ranges = self.__lidar_sensor.getRangeImage()
        lidar_ranges = self.encode_lidar_readings(lidar_ranges)

        delta_lidar_range = lidar_ranges - last_lidar_range

        robot_pos = getPosition(self.robot_node)
        robot_heading = getRotation(self.robot_node)

        target = self.planner.getCurrentTarget()
        goal_heading = np.arctan2(target[1] - robot_pos[1], target[0] - robot_pos[0])
        heading_diff = goal_heading - robot_heading

        local_target = self.planner.getTargetLocal()
        local_robot_pos, local_robot_heading = self.planner.globalToLocal(robot_pos, robot_heading)

        self.obs = np.array([
            linear_velocity/self.v_high,
            angular_velocity/self.w_high,
            local_robot_pos[0]/self.bounds_threshold,
            local_robot_pos[1]/self.bounds_threshold,
            np.sin(local_robot_heading),
            np.cos(local_robot_heading),
            np.sin(heading_diff),
            np.cos(heading_diff),
        ], dtype=np.float32)
        self.obs = np.concatenate((self.obs, delta_lidar_range, lidar_ranges))

        # Termination
        goal_reached = local_robot_pos[0] >= local_target[0] - self.collision_threshold
        final_goal_reached = goal_reached and self.planner.isFinalTarget

        collision = self.obstacleManager.checkCollision(robot_pos[0], robot_pos[1])
        out_of_bounds = (
            abs(local_robot_pos[0]) > self.bounds_threshold or
            abs(local_robot_pos[1]) > self.bounds_threshold
        )
        
        terminated = bool(
            collision or
            (out_of_bounds and self.truncation) or  # out of bounds
            final_goal_reached
        )

        # print("Is terminated?", terminated) #TEST

        # Truncated
        truncated = self.truncation and self.step_lapsed >= self.spec.max_episode_steps
        self.step_lapsed += 1

        if goal_reached or (out_of_bounds and not self.truncation):
            self.planner.project_next_target(robot_pos)
            target = self.planner.getCurrentTarget()
            setPosition(self.goal_node, target)

        # Reward
        reward = self.reward_func(self.obs)
        if collision:
            reward -= 100.0
        elif goal_reached:
            reward += 30.0
        elif out_of_bounds or truncated:
            reward -= 30.0

        reward -= 0.01 * self.step_lapsed  # time penalty

        return (self.obs.astype(np.float32),
                reward, terminated, truncated,
                {
                    'success': final_goal_reached,
                    'collision': collision,
                    'out_of_bounds': out_of_bounds,
                    'goal_heading': goal_heading,
                }
               )
    
    def drawLines(self, lines):
        self.lineManager.drawLines(lines)
