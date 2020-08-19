properties = []
defaults = {}
particle_setup = []
grid_config = []
ncells = []
blocks = []
produced_stmts = []
nprops = 0
ntimesteps = 0

from ast import BlockAST, ExprAST, IfAST, IterAST, NbIterAST
from block_types import ParticlePairsBlock, ParticlesBlock

class Property:
    def __init__(self, prop_name, default_value, volatile):
        self.prop_name = prop_name
        self.default_value = default_value
        self.volatile = volatile

    def __getitem__(self, expr_ast):
        return ExprAST(self.prop_name, expr_ast, '[]', True)

def add_real_property(prop_name, value=0.0, volatile=False):
    return add_property(prop_name, value)

def add_vector_property(prop_name, value=[0.0, 0.0, 0.0], volatile=False):
    return add_property(prop_name, value)

def add_property(prop_name, value, volatile=False):
    prop = Property(prop_name, value, volatile)
    properties.append(prop)
    return prop

def setup_grid(config):
    global grid_config
    grid_config = config

def create_particle_lattice(config, spacing, props={}):
    nx = int((config[0][1] - config[0][0]) / spacing[0]) + 1
    ny = int((config[1][1] - config[1][0]) / spacing[1]) + 1
    nz = int((config[2][1] - config[2][0]) / spacing[2]) + 1

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

                particle_setup.append(particle_props)

def setup_cell_lists(cutoff_radius):
    global grid_config

    ncells = [
        (grid_config[0][1] - grid_config[0][0]) / cutoff_radius,
        (grid_config[1][1] - grid_config[1][0]) / cutoff_radius,
        (grid_config[2][1] - grid_config[2][0]) / cutoff_radius
    ]

def set_timesteps(ts):
    ntimesteps = ts

def particle_pairs(cutoff_radius=None, position=None):
    global blocks
    global produced_stmts

    i = IterAST()
    j = NbIterAST()
    block_stmts = []

    if cutoff_radius is not None and position is not None:
        delta = position[i] - position[j]
        rsq = vector_len_sq(delta)
        yield i, j, delta, rsq
        block_stmts.append(IfAST(rsq < cutoff_radius, produced_stmts.copy(), None))

    else:
        yield i, j
        block_stmts.append(produced_stmts.copy())

    blocks.append(BlockAST(block_stmts, ParticlePairsBlock))
    #produced_stmts = []

def particles():
    global produced_stmts
    global blocks

    yield IterAST()
    blocks.append(BlockAST(produced_stmts.copy(), ParticlesBlock))
    #produced_stmts = []

def vector_len_sq(expr):
    return ExprAST(expr, None, 'vector_len_sq')

def generate():
    global blocks
    for block in blocks:
        block.generate()
