import numpy as np

class ObstacleConfig:
    def __init__(self, count,
                 collision_threshold=0.0,
                 x_range=(0,0), y_range=(0,0),
                 x_speed_range=(0,0), y_speed_range=(0,0),
                 period_range=(100,200)):
        self.count = count
        self.x_range = x_range
        self.y_range = y_range
        self.x_speed_range = x_speed_range
        self.y_speed_range = y_speed_range
        self.period_range = period_range
        self.collision_threshold = collision_threshold

setPosition = lambda node, x, y: node.getField('translation').setSFVec3f([x, y, 0.0])

###################### Obstacle Manager Classes ######################

class ObstacleManager:
    #red cylindar obstacle
    protoString = """
        Solid {
            translation 0 0 0
            rotation 0 0 1 0
            children [
                Shape {
                    appearance PBRAppearance {
                        baseColor 1 0 0
                    }
                    geometry Cylinder {
                        radius 0.1
                        height 1
                    }
                }
            ]
        }
    """
    
    def __init__(self, root_children, config):
        root_children.importMFNodeFromString(-1, "DEF OBSTACLES Group { }")
        group = root_children.getMFNode(root_children.getCount() - 1)
        self.field = group.getField("children")
        self.obstacles = []
        self.config = config
        self.reset()

    def setConfig(self, obstacle_config):
        self.config = obstacle_config
        self.reset()

    def reset(self):
        #cannot remove from group
        for _ in range(len(self.obstacles)): #tear down all obstacles
            self.obstacles.pop()
            self.field.removeMF(0)

        #if obstacle array less than count, add more
        for _ in range(self.config.count):
            self.field.importMFNodeFromString(-1, ObstacleManager.protoString)
            node = self.field.getMFNode(self.field.getCount() - 1)
            obstacle_info = ObstacleInfo()
            self.obstacles.append((node, obstacle_info))

        for node, info in self.obstacles:
            x,y = info.randomise(self.config)
            setPosition(node, x, y)

    def step(self):
        for node, info in self.obstacles:
            reset, x, y = info.step()
            setPosition(node, x, y)
            if reset:
                info.randomiseSpeed(self.config)

    def checkCollision(self, robot_x, robot_y):
        for node, info in self.obstacles:
            obs_x, obs_y = info.get_position()
            dist = np.sqrt((robot_x - obs_x)**2 + (robot_y - obs_y)**2)
            if dist < self.config.collision_threshold:
                return True
        return False


class ObstacleInfo:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.speed_v = 0
        self.speed_w = 0
        self.periodicity = 0
        self.time = 0

    def randomise(self, config):
        self.x = np.random.uniform(config.x_range[0], config.x_range[1])
        self.y = np.random.uniform(config.y_range[0], config.y_range[1])
        #if x y is within 0.5 of origin, re-randomise, max 10 times
        count = 0
        while np.sqrt(self.x**2 + self.y**2) < 0.5 or count < 10:
            self.x = np.random.uniform(config.x_range[0], config.x_range[1])
            self.y = np.random.uniform(config.y_range[0], config.y_range[1])
            count += 1
        self.randomiseSpeed(config)
        return self.x, self.y

    def randomiseSpeed(self, config):
        self.x_speed = np.random.uniform(config.x_speed_range[0], config.x_speed_range[1]) * 0.0032
        self.y_speed = np.random.uniform(config.y_speed_range[0], config.y_speed_range[1]) * 0.0032
        self.periodicity = np.random.randint(config.period_range[0], config.period_range[1])
        self.time = 0

    def step(self):
        self.x += self.x_speed
        self.y += self.y_speed
        self.time += 1
        return self.time >= self.periodicity,self.x, self.y

    def get_position(self):
        return self.x, self.y

    def set_position(self, x, y):
        self.x, self.y = x, y

    
