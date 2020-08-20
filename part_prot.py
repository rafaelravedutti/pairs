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
from properties import Property
from block_types import ParticlePairsBlock, ParticlesBlock
from printer import printer

def add_real_property(prop_name, value=0.0, volatile=False):
    return add_property(prop_name, 'real', value)

def add_vector_property(prop_name, value=[0.0, 0.0, 0.0], volatile=False):
    return add_property(prop_name, 'vector', value)

def add_property(prop_name, prop_type, value, volatile=False):
    prop = Property(prop_name, prop_type, value, volatile)
    properties.append(prop)
    defaults[prop_name] = value
    return prop

def setup_grid(config):
    global grid_config
    grid_config = config

def create_particle_lattice(config, spacing, props={}):
    nx = int((config[0][1] - config[0][0]) / spacing[0] - 0.001) + 1
    ny = int((config[1][1] - config[1][0]) / spacing[1] - 0.001) + 1
    nz = int((config[2][1] - config[2][0]) / spacing[2] - 0.001) + 1

    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):
                particle_props = defaults.copy()

                for p in props:
                    particle_props[p] = props[p]

                particle_props['position'] = [
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

def generate_properties_decl():
    for p in properties:
        if p.prop_type == 'real':
            printer.print("double {}[{}];".format(p.prop_name, len(particle_setup)))
        elif p.prop_type == 'vector':
            printer.print("double {}[{}][3];".format(p.prop_name, len(particle_setup)))
        else:
            raise Exception("Invalid property type!")

def generate_setup():
    index = 0
    for ps in particle_setup:
        for key in ps:
            vname = "{}[{}]".format(key, index)
            if isinstance(ps[key], list):
                output =  "{}[0] = {}, ".format(vname, ps[key][0])
                output += "{}[1] = {}, ".format(vname, ps[key][1])
                output += "{}[2] = {};".format(vname, ps[key][2])
                printer.print(output)
            else:
                printer.print("{} = {};".format(vname, ps[key]))

        index += 1

def generate():
    printer.print("int main() {")
    printer.add_ind(4)
    printer.print("const int nparticles = {};".format(len(particle_setup)))
    generate_properties_decl()
    generate_setup()
    global blocks
    for block in blocks:
        block.generate()
    printer.add_ind(-4)
    printer.print("}")
