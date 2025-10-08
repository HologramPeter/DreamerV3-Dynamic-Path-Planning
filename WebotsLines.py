class LineManager:
    protoString = """
        DEF LineShape Shape {
            geometry IndexedLineSet {
                coord Coordinate {
                point [
                    %s
                ]
                }
                coordIndex [
                %s
                ]
            }
            appearance Appearance {
                material Material {
                emissiveColor %s
                }
            }
        }
    """

    def __init__(self, root_children):
        root_children.importMFNodeFromString(-1, "DEF LINES Group { }")
        group = root_children.getMFNode(root_children.getCount() - 1)
        self.field = group.getField("children")
        self.line_count = 0

    def reset(self):
        while self.line_count > 0:
            self.field.removeMF(0)
            self.line_count -= 1

    def drawLines(self, lines):
        self.reset()

        for isGreen, line in lines:
            point_str = ', '.join(['%f %f %f' % tuple(p) for p in line])
            index_str = ', '.join(str(i) for i in range(len(line)))# + ', -1'
            line_node_string = LineManager.protoString % (point_str, index_str, '1 0 0' if isGreen else '0 1 0')
            self.field.importMFNodeFromString(-1, line_node_string)
            self.line_count += 1