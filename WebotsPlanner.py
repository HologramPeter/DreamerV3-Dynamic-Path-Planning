#creates a transform planner object
# storing the target position
"""
transform planner class

stores:
- global_target
- planned_target
- planned_target_distance
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

class Transform:
    def __init__(self, position, heading):
        self.position = position # np.array([x, y])
        self.heading = heading # in radians

class PlannerConfig:
    def __init__(self, target, half_size):
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

    def __init__(self, root_children, config: PlannerConfig):
        root_children.importMFNodeFromString(-1, "DEF PLANNER Group { }")
        group = root_children.getMFNode(root_children.getCount() - 1)
        self.field = group.getField("children")
        self.isDisplayed = False

        self.current_transform = Transform(np.array([0.0, 0.0]), 0.0)
        self.global_target = None
        self.planned_target = None
        self.half_size = None
        self.setConfig(config)

    def setConfig(self, config: PlannerConfig):
        self.global_target = config.target
        self.half_size = config.half_size
        self.project_next_target()

    def set_transform(self, transform):
        self.current_transform = transform
        self.project_next_target()

    def globalToLocal(self, global_point):
        #transform a global point to local coordinates
        offset = global_point - self.current_transform.position
        c = np.cos(-self.current_transform.heading)
        s = np.sin(-self.current_transform.heading)
        rotation_matrix = np.array([[c, -s], [s, c]])
        local_point = rotation_matrix.dot(offset)
        return local_point
    
    def localToGlobal(self, local_point):
        #transform a local point to global coordinates
        c = np.cos(self.current_transform.heading)
        s = np.sin(self.current_transform.heading)
        rotation_matrix = np.array([[c, -s], [s, c]])
        global_point = rotation_matrix.dot(local_point) + self.current_transform.position
        return global_point
    
    def globalToLocalHeading(self, global_heading):
        #transform a global heading to local coordinates
        return global_heading - self.current_transform.heading

    def localToGlobalHeading(self, local_heading):
        #transform a local heading to global coordinates
        return local_heading + self.current_transform.heading
    
    def project_next_target(self):
        #project the next target based on current transform and global target
        target_vector = self.global_target - self.current_transform.position
        distance = np.linalg.norm(target_vector)
        if distance < self.half_size:
            self.planned_target = self.global_target
        else:
            target_vector = target_vector / distance  # normalize
            self.planned_target = self.current_transform.position + target_vector * self.half_size
        self.draw_square()

    def draw_square(self):
        if self.isDisplayed:
            self.field.removeMF(0)
            self.isDisplayed = False
        #draw a square around the current transform
        half_size = self.planned_target_distance / 2
        corners = np.array([
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size]
        ])
        global_corners = [self.localToGlobal(corner) for corner in corners]
        point_str = ', '.join(['%f %f %f' % (p[0], p[1], 0) for p in global_corners])
        line_node_string = Planner.protoString % (point_str)
        return line_node_string
    
    