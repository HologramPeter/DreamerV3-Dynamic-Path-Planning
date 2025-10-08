import numpy as np

class RewardGenerator:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.target = np.array([1.0, 0.0])
        #angle of each of n sectors, rather than 4
        sectors = 16
        sector_increment = 2 * np.pi / sectors
        func = lambda i: i * sector_increment
        self.sector_angles = np.array([func(i) for i in range(sectors)])

    def setVerbose(self, verbose):
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
        heading_sin, heading_cos = obs[4:6]  # cos of robot heading
        heading_diff_sin, heading_diff_cos = obs[6:8]  # cos, sin of robot heading relative to goal
        obstacle_info = obs[8:].reshape(-1, 2)  # angle, distance of obstacles

        # heading = np.arctan2(heading_sin, heading_cos) / np.pi * 180
        # heading_diff = np.arctan2(heading_diff_sin, heading_diff_cos) / np.pi * 180
        #round to 2 decimal places
        # heading = round(heading, 2)
        # heading_diff = round(heading_diff, 2)
        # self._verbose(f"Heading: {heading}, Heading diff: {heading_diff}")

        dist_to_goal = np.linalg.norm(robot_pos - self.target)

        heading_reward = heading_diff_cos

        # index_to_closest_obstacle = np.argmin(dist_to_obstacles)
        # dist_to_obstacles = dist_to_obstacles

        obstacle_reward = 1/((obstacle_info[:, 1]*5)**2 + 0.01) #cap this to 0.01

        # goal_reward = 1/((dist_to_goal*2) + 0.01) #exp #cap this to 0.01
        if dist_to_goal > 0.4:
            goal_reward = -dist_to_goal**2
        elif dist_to_goal < 0.2:
            goal_reward = 1/((dist_to_goal*5)**2 + 0.01) #cap this to 0.01
        else:
            goal_reward = 0.0

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
        #function to map angle to sector multiplier
        func = lambda angle: max(np.cos(angle), 0) + max(np.cos(angle + np.pi/2), 0)
        self.sectors_mul = np.array([func(angle) for angle in self.sector_angles])

    def compute(self, linear_velocity, angular_velocity, dist_to_goal,
                heading_reward, obstacle_reward, goal_reward):

        obstacle_reward = np.mean(obstacle_reward * self.sectors_mul)
        if linear_velocity < 0:
            # obstacle_reward = np.sum(obstacle_reward * self.reverse_sectors_mul)
            speed_reward = 0
            reverse_penalty = 1
        else:
            # obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)
            reverse_penalty = 0
            if linear_velocity < 0.05:
                speed_reward = 0.1
            else:
                speed_reward = 0.2

        #add reward for turning with slow linear velocity
        if linear_velocity < 0.05:
            angular_reward = 0
        else:
            angular_reward = max(abs(angular_velocity) - 2.0, 0)

        reward = (
            - 15.0 * dist_to_goal
            + 10.0 * goal_reward
            - 0.5 * obstacle_reward
            + 0.3 * speed_reward
            - 0.3 * angular_reward
            - 5.0 * reverse_penalty
            + 6.0 * speed_reward * heading_reward
        )

        self._verbose(
              f"Reward: {reward:>10.4f}, "
              f"Velocity: {linear_velocity:>6.3f}, "
              f"Heading Reward: {heading_reward:>8.4f}, "
              f"Distance to Goal: {dist_to_goal:>8.4f}, "
              f"Obstacle Reward: {obstacle_reward:>8.4f}, "
            )
        return reward
    
class RewardGeneratorLeftTurn(RewardGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        #function to map angle to sector multiplier
        func = lambda angle: np.cos(angle) + np.cos(angle - np.pi/2)
        self.sectors_mul = np.array([func(angle) for angle in self.sector_angles])

    def compute(self, linear_velocity, angular_velocity, dist_to_goal,
                heading_reward, obstacle_reward, goal_reward):
        obstacle_reward = np.mean(obstacle_reward * self.sectors_mul)
        if linear_velocity < 0:
            # obstacle_reward = np.sum(obstacle_reward * self.reverse_sectors_mul)
            speed_reward = -0.5
        else:
            # obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)
            if linear_velocity < 0.05:
                speed_reward = 0.1
            else:
                speed_reward = 0.2

        #add reward for turning with slow linear velocity
        if linear_velocity < 0.05:
            angular_reward = 0
        else:
            angular_reward = max(abs(angular_velocity) - 2.0, 0)

        reward = (
            - 9.0 * dist_to_goal
            + 0.5 * goal_reward
            - 0.5 * obstacle_reward
            + 0.3 * speed_reward
            - 0.3 * angular_reward
            + 6.0 * speed_reward * heading_reward
        )

        self._verbose(
              f"Reward: {reward:>10.4f}, "
              f"Obstacle Reward: {obstacle_reward:>8.4f}, "
              f"Heading Reward: {heading_reward:>8.4f}, "
            )
        return reward
    
class RewardGeneratorSteering(RewardGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        # hard code lidar range sectors
        self.sectors_mul = np.array([
            0.2,   # back
            0.4,   # left
            1.0,   # front
            0.4,   # right
        ])

    def compute(self, linear_velocity, angular_velocity, dist_to_goal,
                heading_reward, obstacle_reward, goal_reward):

        obstacle_reward = np.mean(obstacle_reward * self.sectors_mul)
        if linear_velocity < 0:
            # obstacle_reward = np.sum(obstacle_reward * self.reverse_sectors_mul)
            speed_reward = -0.5
        else:
            # obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)
            if linear_velocity < 0.05:
                speed_reward = 0.1
            else:
                speed_reward = 0.2

        #add reward for turning with slow linear velocity
        if linear_velocity < 0.05:
            angular_reward = 0
        else:
            angular_reward = max(abs(angular_velocity) - 2.0, 0)

        reward = (
            - 9.0 * dist_to_goal
            + 0.5 * goal_reward
            - 0.5 * obstacle_reward
            + 0.3 * speed_reward
            - 0.3 * angular_reward
            + 6.0 * speed_reward * heading_reward
        )

        self._verbose(
              f"Reward: {reward:>10.4f}, "
              f"Distance to Goal: {dist_to_goal:>8.4f}, "
              f"Obstacle Reward: {obstacle_reward:>8.4f}, "
            )
        return reward
    
class RewardGeneratorDreamer(RewardGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        #function to map angle to sector multiplier
        func = lambda angle: np.cos(angle)
        self.sectors_mul = np.array([func(angle) for angle in self.sector_angles])

    def compute(self, linear_velocity, angular_velocity, dist_to_goal,
                heading_reward, obstacle_reward, goal_reward):

        obstacle_reward = np.mean(obstacle_reward * self.sectors_mul)
        if linear_velocity < 0:
            # obstacle_reward = np.sum(obstacle_reward * self.reverse_sectors_mul)
            speed_reward = -0.5
        else:
            # obstacle_reward = np.sum(obstacle_reward * self.sectors_mul)
            if linear_velocity < 0.05:
                speed_reward = 0.1
            else:
                speed_reward = 0.2

        #add reward for turning with slow linear velocity
        if linear_velocity < 0.05:
            angular_reward = 0
        else:
            angular_reward = max(abs(angular_velocity) - 2.0, 0)

        reward = (
            - 9.0 * dist_to_goal
            + 0.5 * goal_reward
            - 0.5 * obstacle_reward
            + 0.3 * speed_reward
            - 0.3 * angular_reward
            + 6.0 * speed_reward * heading_reward
        )

        self._verbose(
              f"Reward: {reward:>10.4f}, "
              f"Distance to Goal: {dist_to_goal:>8.4f}, "
              f"Obstacle Reward: {obstacle_reward:>8.4f}, "
            )
        return reward
