#creates a transform planner object
# storing the target position
"""
transform planner class

stores:
- current_target
- current_transform
    - x, y position
    - heading
-functions:
    - set transform
    - creating the next planned target based on current transform and global target
    - transforming global coordinates to local coordinates
    - transforming local coordinates to global coordinates
    - transforming a heading from global to local
    - transforming a heading from local to global
    - drawing a proto square around current_transform
"""

import numpy as np

class PlannerConfig:
    def __init__(self, start: np.ndarray, target: np.ndarray, half_size: float):
        self.start = start # np.array([x, y])
        self.target = target # np.array([x, y])
        self.half_size = half_size # float

class Planner:
    protoString = """
        DEF LocalArea Shape {
            geometry IndexedLineSet {
                coord Coordinate {
                point [
                    %s
                ]
                }
                coordIndex [
                    0 1,
                    1 2,
                    2 3,
                    3 0
                ]
            }
            appearance Appearance {
                material Material {
                emissiveColor 0 0 1
                }
            }
        }
    """

    def __init__(self, config: PlannerConfig, root_children = None):
        if root_children is not None:
            root_children.importMFNodeFromString(-1, "DEF PLANNER Group { }")
            group = root_children.getMFNode(root_children.getCount() - 1)
            self.field = group.getField("children")
        else:
            self.field = None
        self.isDisplayed = False
        self.isFinalTarget = False

        self.current_target: np.ndarray = None
        self.current_position: np.ndarray = None
        self.current_heading: float = None
        self.setConfig(config)

    def setConfig(self, config: PlannerConfig):
        self.config = config
        self.reset()

    def reset(self):
        # print("Planner reset to start", self.config.start, "target", self.config.target) #TEST
        self.isFinalTarget = False
        self.project_next_target(self.config.start)

    def set_position(self, position):
        self.project_next_target(position)

    def localToGlobal(self, position, heading):
        return (
            self.localToGlobalPosition(position),
            self.localToGlobalHeading(heading)
        )

    def globalToLocal(self, position, heading):
        return (
            self.globalToLocalPosition(position),
            self.globalToLocalHeading(heading)
        )

    def globalToLocalPosition(self, global_point):
        #transform a global point to local coordinates
        offset = global_point - self.current_position
        c = np.cos(-self.current_heading)
        s = np.sin(-self.current_heading)
        rotation_matrix = np.array([[c, -s], [s, c]])
        local_point = rotation_matrix.dot(offset)
        return local_point
    
    def localToGlobalPosition(self, local_point):
        #transform a local point to global coordinates
        c = np.cos(self.current_heading)
        s = np.sin(self.current_heading)
        rotation_matrix = np.array([[c, -s], [s, c]])
        global_point = rotation_matrix.dot(local_point) + self.current_position
        return global_point
    
    def globalToLocalHeading(self, global_heading):
        return global_heading - self.current_heading

    def localToGlobalHeading(self, local_heading):
        #transform a local heading to global coordinates
        return local_heading + self.current_heading
    
    def project_next_target(self, position):
        if self.isFinalTarget:
            return
        #project the next target based on current transform and global target
        target_vector = self.config.target - position
        #calculate heading
        heading = np.arctan2(target_vector[1], target_vector[0])

        self.current_position = position
        self.current_heading = heading

        distance = np.linalg.norm(target_vector)
        self.isFinalTarget = distance < self.config.half_size+0.0001
        if self.isFinalTarget:
            self.current_target = self.config.target
        else:
            target_vector = target_vector / distance  # normalize
            self.current_target = self.current_position + target_vector * self.config.half_size

        #print position-target, distance, heading, current target
        # print("Planner:", position, "to", self.config.target, ", Dist", distance, ",heading", heading, ",current target", self.current_target)  #TEST

        self.draw_square()

    def draw_square(self):
        if self.field is None:
            return
        if self.isDisplayed:
            self.field.removeMF(0)
            self.isDisplayed = False
        #draw a square around the current transform
        corners = np.array([
            [-self.config.half_size, -self.config.half_size],
            [self.config.half_size, -self.config.half_size],
            [self.config.half_size, self.config.half_size],
            [-self.config.half_size, self.config.half_size]
        ])
        global_corners = [self.localToGlobalPosition(corner) for corner in corners]
        point_str = ', '.join(['%f %f %f' % (p[0], p[1], 0) for p in global_corners])
        line_node_string = Planner.protoString % (point_str)
        self.field.importMFNodeFromString(-1, line_node_string)
        self.isDisplayed = True
    
    def getTransform(self):
        return self.current_position, self.current_heading
    
    def getCurrentTarget(self):
        return self.current_target
    
    def getTargetLocal(self):
        return self.globalToLocalPosition(self.current_target)
