import numpy as np

class RewardGenerator:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def _verbose(self, msg):
        if self.verbose:
            print(msg)

    def __call__(self, obs):
        return self.reward(obs)
    
    def reward(self, obs):
        #input shape: (14,)

        linear_velocity = obs[0]  # linear velocity
        angular_velocity = obs[1]  # angular velocity
        robot_pos = obs[2:4]  # robot x, y position
        # heading_sin = obs[4]  # sin, cos of robot heading relative to goal
        heading_cos = obs[5]  # sin, cos of robot heading relative to goal
        dist_to_obstacles = obs[6:10]  # lidar readings


        linear_velocity = obs[0]  # linear velocity
        angular_velocity = obs[1]  # angular velocity
        robot_pos = obs[2:4]  # robot x, y position
        # heading_sin = obs[4]  # sin, cos of robot heading relative to goal
        heading_cos = obs[5]  # sin, cos of robot heading relative to goal
        dist_to_obstacles = obs[6:10]  # lidar readings

        dist_to_goal = (np.linalg.norm(robot_pos)*2) ** 2 # scaled down base on environment size

        heading_reward = (heading_cos + 1) / 2

        obstacle_reward = 1/((dist_to_obstacles*10)**2 +0.001)
        goal_reward = 1/(dist_to_goal*10 +0.001)

        return self.compute(
            linear_velocity, angular_velocity, dist_to_goal,
            heading_reward, obstacle_reward, goal_reward
        )

    def compute(self, linear_velocity, angular_velocity, dist_to_goal,
                heading_reward, obstacle_reward, goal_reward):
        return 0


class RewardGeneratorRightTurn(RewardGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        # hard code lidar range sectors
        self.sectors_mul = np.array([
            1.0,   # front
            0.5,   # left
            0.2,   # right
            0.2,   # back
        ])
        # self.reverse_sectors_mul = np.array([
        #     0.2,   # front
        #     0.2,   # left
        #     0.5,   # right
        #     1.0,   # back
        # ])

    def compute(self, linear_velocity, angular_velocity, dist_to_goal,
                heading_reward, obstacle_reward, goal_reward):
        
        obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)
        if linear_velocity < 0:
            # obstacle_reward = np.sum(obstacle_reward * self.reverse_sectors_mul)
            speed_reward = -0.5
        else:
            # obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)
            if linear_velocity < 0.05:
                speed_reward = 0.1
            else:
                speed_reward = 0.2

        #add reward for turning with no linear velocity
        if linear_velocity == 0:
            angular_reward = -1.2 * abs(angular_velocity)
        else:
            angular_reward = max(abs(angular_velocity) - 2.0, 0)

        reward = (
            - 20.0 * dist_to_goal
            + 05.0 * goal_reward
            - 05.0 * obstacle_reward
            + 04.0 * speed_reward
            - 04.0 * angular_reward
            + 30.0 * speed_reward * heading_reward
        )

        self._verbose(
              f"Reward: {reward:>10.4f}, "
              f"Distance to Goal: {dist_to_goal:>8.4f}, "
              f"Obstacle Reward: {obstacle_reward:>8.4f}, "
              f"Heading Reward: {heading_reward:>8.4f}, "

            #   f"Front Lidar: {dist_to_obstacles[0]:>8.4f}, "
            #   f"Left Lidar: {dist_to_obstacles[1]:>8.4f}, "
            #   f"Right Lidar: {dist_to_obstacles[2]:>8.4f}, "
            #   f"Back Lidar: {dist_to_obstacles[3]:>8.4f}"

            #   f"linear_velocity: {linear_velocity:>10.4f}, "
            #   f"angular_velocity: {angular_velocity:>10.4f}, "
            )
        return reward

class RewardGeneratorLeftTurn(RewardGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        # hard code lidar range sectors
        self.sectors_mul = np.array([
            1.0,   # front
            0.2,   # left
            0.5,   # right
            0.2,   # back
        ])
        # self.reverse_sectors_mul = np.array([
        #     0.2,   # front
        #     0.5,   # left
        #     0.2,   # right
        #     1.0,   # back
        # ])

    def compute(self, linear_velocity, angular_velocity, dist_to_goal,
                heading_reward, obstacle_reward, goal_reward):
        
        obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)
        if linear_velocity < 0:
            # obstacle_reward = np.sum(obstacle_reward * self.reverse_sectors_mul)
            speed_reward = -0.5
        else:
            # obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)
            if linear_velocity < 0.05:
                speed_reward = 0.1
            else:
                speed_reward = 0.2

        #add reward for turning with no linear velocity
        if linear_velocity == 0:
            angular_reward = -1.2 * abs(angular_velocity)
        else:
            angular_reward = max(abs(angular_velocity) - 2.0, 0)

        reward = (
            - 20.0 * dist_to_goal
            + 05.0 * goal_reward
            - 05.0 * obstacle_reward
            + 04.0 * speed_reward
            - 04.0 * angular_reward
            + 30.0 * speed_reward * heading_reward
        )

        self._verbose(
              f"Reward: {reward:>10.4f}, "
              f"Distance to Goal: {dist_to_goal:>8.4f}, "
              f"Obstacle Reward: {obstacle_reward:>8.4f}, "
              f"Heading Reward: {heading_reward:>8.4f}, "

            #   f"Front Lidar: {dist_to_obstacles[0]:>8.4f}, "
            #   f"Left Lidar: {dist_to_obstacles[1]:>8.4f}, "
            #   f"Right Lidar: {dist_to_obstacles[2]:>8.4f}, "
            #   f"Back Lidar: {dist_to_obstacles[3]:>8.4f}"

            #   f"linear_velocity: {linear_velocity:>10.4f}, "
            #   f"angular_velocity: {angular_velocity:>10.4f}, "
            )
        return reward
    
class RewardGeneratorSteering(RewardGenerator):
    def compute(self, linear_velocity, angular_velocity, dist_to_goal,
                heading_reward, obstacle_reward, goal_reward):
        
        speed_reward = 0.0
        if linear_velocity < 0:
            speed_reward = -0.5
        elif linear_velocity < 0.05:
            speed_reward = 0.1
        else:
            speed_reward = 0.2

        #hard code lidar range sectors
        sectors_mul = np.array([
            1.0,   # front
            0.4,   # left
            0.4,   # right
            0.2,   # back
        ])
        obstacle_reward = np.sum(sectors_mul*obstacle_reward)

        angular_reward = max(abs(angular_velocity) - 2.0, 0)

        reward = (
            - 20.0 * dist_to_goal
            + 05.0 * goal_reward
            - 05.0 * obstacle_reward
            + 04.0 * speed_reward
            - 04.0 * angular_reward
            + 20.0 * speed_reward * heading_reward
        )

        self._verbose(
              f"Reward: {reward:>10.4f}, "
              f"Distance to Goal: {dist_to_goal:>8.4f}, "
              f"Obstacle Reward: {obstacle_reward:>8.4f}, "
              f"Heading Reward: {heading_reward:>8.4f}, "

            #   f"Front Lidar: {dist_to_obstacles[0]:>8.4f}, "
            #   f"Left Lidar: {dist_to_obstacles[1]:>8.4f}, "
            #   f"Right Lidar: {dist_to_obstacles[2]:>8.4f}, "
            #   f"Back Lidar: {dist_to_obstacles[3]:>8.4f}"

            #   f"linear_velocity: {linear_velocity:>10.4f}, "
            #   f"angular_velocity: {angular_velocity:>10.4f}, "
            )
        return reward

class RewardGeneratorStraight(RewardGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        # hard code lidar range sectors
        self.sectors_mul = np.array([
            0.8,   # front
            0.4,   # left
            0.4,   # right
            0.2,   # back
        ])
        # self.reverse_sectors_mul = np.array([
        #     0.2,   # front
        #     0.5,   # left
        #     0.2,   # right
        #     1.0,   # back
        # ])

    def compute(self, linear_velocity, angular_velocity, dist_to_goal,
                heading_reward, obstacle_reward, goal_reward):
        
        obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)


        # NOTE: try this next time
        # if linear_velocity < 0:
        #     # obstacle_reward = np.sum(obstacle_reward * self.reverse_sectors_mul)
        #     speed_reward = 0.2
        # elif linear_velocity == 0:
        #     speed_reward = 0.1
        # elif linear_velocity < 0.15:
        #     speed_reward = 0.00
        # else:
        #     speed_reward = 0.2
        
        # #add reward for turning with no linear velocity
        # if linear_velocity == 0:
        #     angular_reward = -1.2 * abs(angular_velocity)
        # else:
        #     angular_reward = 3 * abs(angular_velocity)

        if linear_velocity <= 0:
            # obstacle_reward = np.sum(obstacle_reward * self.reverse_sectors_mul)
            speed_reward = 0.1
        else:
            # obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)
            if linear_velocity < 0.05:
                speed_reward = 0.1
            else:
                speed_reward = 0.2

        angular_reward = 3 * abs(angular_velocity)

        #only give heading reward if it is 1
        if heading_reward < 0.9:
            heading_reward = 0.0

        reward = (
            - 15.0 * dist_to_goal
            + 05.0 * goal_reward
            - 20.0 * obstacle_reward
            + 04.0 * speed_reward
            - 04.0 * angular_reward
            + 10.0 * heading_reward
        )

        self._verbose(
              f"Reward: {reward:>10.4f}, "
              f"Distance to Goal: {dist_to_goal:>8.4f}, "
              f"Obstacle Reward: {obstacle_reward:>8.4f}, "
              f"Heading Reward: {heading_reward:>8.4f}, "

            #   f"Front Lidar: {dist_to_obstacles[0]:>8.4f}, "
            #   f"Left Lidar: {dist_to_obstacles[1]:>8.4f}, "
            #   f"Right Lidar: {dist_to_obstacles[2]:>8.4f}, "
            #   f"Back Lidar: {dist_to_obstacles[3]:>8.4f}"

            #   f"linear_velocity: {linear_velocity:>10.4f}, "
            #   f"angular_velocity: {angular_velocity:>10.4f}, "
            )
        return reward
    

class RewardGeneratorStraight2(RewardGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        # hard code lidar range sectors
        self.sectors_mul = np.array([
            0.8,   # front
            0.4,   # left
            0.4,   # right
            0.2,   # back
        ])
        # self.reverse_sectors_mul = np.array([
        #     0.2,   # front
        #     0.5,   # left
        #     0.2,   # right
        #     1.0,   # back
        # ])

    def compute(self, linear_velocity, angular_velocity, dist_to_goal,
                heading_reward, obstacle_reward, goal_reward):
        
        obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)


        # NOTE: try this next time
        if linear_velocity < 0:
            # obstacle_reward = np.sum(obstacle_reward * self.reverse_sectors_mul)
            speed_reward = 0.0
        elif linear_velocity == 0:
            speed_reward = 0.1
        elif linear_velocity < 0.15:
            speed_reward = 0.00
        else:
            speed_reward = 0.2
        
        #add reward for turning with no linear velocity
        if linear_velocity == 0:
            angular_reward = -1.2 * abs(angular_velocity)
        else:
            angular_reward = 3 * abs(angular_velocity)

        # if linear_velocity <= 0:
        #     # obstacle_reward = np.sum(obstacle_reward * self.reverse_sectors_mul)
        #     speed_reward = 0.1
        # else:
        #     # obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)
        #     if linear_velocity < 0.05:
        #         speed_reward = 0.1
        #     else:
        #         speed_reward = 0.2

        # angular_reward = 3 * abs(angular_velocity)

        #only give heading reward if it is 1
        if heading_reward < 0.9:
            heading_reward = 0.0

        reward = (
            - 15.0 * dist_to_goal
            + 05.0 * goal_reward
            - 20.0 * obstacle_reward
            + 04.0 * speed_reward
            - 04.0 * angular_reward
            + 10.0 * heading_reward
        )

        self._verbose(
              f"Reward: {reward:>10.4f}, "
              f"Distance to Goal: {dist_to_goal:>8.4f}, "
              f"Obstacle Reward: {obstacle_reward:>8.4f}, "
              f"Heading Reward: {heading_reward:>8.4f}, "

            #   f"Front Lidar: {dist_to_obstacles[0]:>8.4f}, "
            #   f"Left Lidar: {dist_to_obstacles[1]:>8.4f}, "
            #   f"Right Lidar: {dist_to_obstacles[2]:>8.4f}, "
            #   f"Back Lidar: {dist_to_obstacles[3]:>8.4f}"

            #   f"linear_velocity: {linear_velocity:>10.4f}, "
            #   f"angular_velocity: {angular_velocity:>10.4f}, "
            )
        return reward