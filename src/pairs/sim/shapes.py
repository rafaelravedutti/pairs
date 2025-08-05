class Shapes:
    Invalid     =  -1
    Sphere      =   0
    Halfspace   =   1
    PointMass   =   2
    Box         =   3
    Clump       =   4

    def name(shape_id):
        return  "Sphere"    if shape_id == Shapes.Sphere else \
                "Halfspace" if shape_id == Shapes.Sphere else \
                "PointMass" if shape_id == Shapes.PointMass else \
                "Box"       if shape_id == Shapes.Box else \
                "Clump"     if shape_id == Shapes.Clump else \
                "Invalid"     

    def id(shape_obj):
        return  Shapes.Sphere       if isinstance(shape_obj, Sphere) else \
                Shapes.Halfspace    if isinstance(shape_obj, Halfspace) else \
                Shapes.PointMass    if isinstance(shape_obj, PointMass) else \
                Shapes.Box          if isinstance(shape_obj, Box) else \
                Shapes.Clump        if isinstance(shape_obj, Clump) else \
                Shapes.Invalid

class Sphere:
    def __init__(self, radius):
        self.radius = radius


class Halfspace:
    def __init__(self, normal):
        self.normal = normal


class PointMass:
    def __init__(self):
        pass

class Box:
    def __init__(self, edge_length, rotation_matrix):
        self.edge_length = edge_length
        self.rotation_matrix = rotation_matrix


class Clump:
    def __init__(self, local_positions, local_radii, rotation_matrix):
        self.local_positions = local_positions
        self.local_radii = local_radii
        self.rotation_matrix = rotation_matrix
        