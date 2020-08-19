properties = {}
defaults = {}
particle_setup = []
grid_config = []
ncells = []
nprops = 0
ntimesteps = 0

def add_real_property(prop_name, value=0.0):
    add_property(prop_name, value)

def add_vector_property(prop_name, value=[0.0, 0.0, 0.0]):
    add_property(prop_name, value)

def add_property(prop_name, value):
    prop = Property(prop_name, value)
    properties.append(prop)
    return prop

def setup_grid(config):
    grid_config = config

def create_particle_lattice(config, spacing, props={}):
    nx = (config[0][1] - config[0][0]) / spacing[0]
    ny = (config[1][1] - config[1][0]) / spacing[1]
    nz = (config[2][1] - config[2][0]) / spacing[2]

    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):
                particle_props = defaults.copy()

                for p in props:
                    particle_props[p] = props[p]

                particle_props['positions'] = [
                    config[0][0] + spacing[0] * i,
                    config[1][0] + spacing[1] * j,
                    config[2][0] + spacing[2] * k
                ]

                particle_setup.insert(particle_props)

def setup_cell_lists(cutoff_radius):
    ncells = [
        (grid_config[0][1] - grid_config[0][0]) / cutoff_radius,
        (grid_config[1][1] - grid_config[1][0]) / cutoff_radius,
        (grid_config[2][1] - grid_config[2][0]) / cutoff_radius
    ]

def set_timesteps(ts):
    ntimesteps = ts

def particle_pairs(cutoff_radius=None, position=None):
    i = IterAST()
    j = NbIterAST()

    if cutoff_radius is not None and position is not None:
        delta = position[i] - position[j]
        rsq = pt.vector_len_sq(delta)
        block.append(IfAST(rsq < cutoff_radius))
        yield i, j, delta, rsq

    else:
        yield i, j

def particles():
    yield IterAST()

def vector_len_sq(expr):
    ExprAST(expr, None, 'vector_len_sq')
