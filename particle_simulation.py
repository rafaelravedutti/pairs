from ast import BlockAST, ExprAST, IfAST, IterAST, NbIterAST
from properties import Property
from block_types import ParticlePairsBlock, ParticlesBlock
from printer import printer

class ParticleSimulation:
    def __init__(self):
        self.properties = []
        self.defaults = {}
        self.setup = []
        self.grid_config = []
        self.blocks = []
        self.produced_stmts = []
        self.ntimesteps = 0
        self.expr_id = 0

    def add_property(self, prop_name, prop_type, value, volatile):
        prop = Property(self, prop_name, prop_type, value, volatile)
        self.properties.append(prop)
        self.defaults[prop_name] = value
        return prop

    def add_real_property(self, prop_name, value=0.0, volatile=False):
        return self.add_property(prop_name, 'real', value, volatile)

    def add_vector_property(self, prop_name, value=[0.0, 0.0, 0.0], volatile=False):
        return self.add_property(prop_name, 'vector', value, volatile)

    def new_expr(self):
        self.expr_id += 1
        return self.expr_id - 1

    def setup_grid(self, config):
        self.grid_config = config

    def create_particle_lattice(self, config, spacing, props={}):
        nx = int((config[0][1] - config[0][0]) / spacing[0] - 0.001) + 1 
        ny = int((config[1][1] - config[1][0]) / spacing[1] - 0.001) + 1 
        nz = int((config[2][1] - config[2][0]) / spacing[2] - 0.001) + 1 

        for i in range(0, nx):
            for j in range(0, ny):
                for k in range(0, nz):
                    particle_props = self.defaults.copy()

                    for p in props:
                        particle_props[p] = props[p]

                    particle_props['position'] = [ 
                        config[0][0] + spacing[0] * i,
                        config[1][0] + spacing[1] * j,
                        config[2][0] + spacing[2] * k 
                    ]

                    self.setup.append(particle_props)

    def setup_cell_lists(self, cutoff_radius):
        ncells = [ 
            (self.grid_config[0][1] - self.grid_config[0][0]) / cutoff_radius,
            (self.grid_config[1][1] - self.grid_config[1][0]) / cutoff_radius,
            (self.grid_config[2][1] - self.grid_config[2][0]) / cutoff_radius
        ]   

    def set_timesteps(self, ts):
        self.ntimesteps = ts

    def particle_pairs(self, cutoff_radius=None, position=None):
        i = IterAST()
        j = NbIterAST()
        block_stmts = []

        if cutoff_radius is not None and position is not None:
            delta = position[i] - position[j]
            rsq = self.vector_len_sq(delta)
            yield i, j, delta, rsq
            block_stmts.append(IfAST(rsq < cutoff_radius, self.produced_stmts.copy(), None))

        else:
            yield i, j
            block_stmts.append(produced_stmts.copy())

        self.blocks.append(BlockAST(block_stmts, ParticlePairsBlock))
        #self.produced_stmts = []

    def particles(self):
        yield IterAST()
        self.blocks.append(BlockAST(self.produced_stmts.copy(), ParticlesBlock))
        #self.produced_stmts = []

    def vector_len_sq(self, expr):
        return ExprAST(self, expr, None, 'vector_len_sq')

    def generate_properties_decl(self):
        for p in self.properties:
            if p.prop_type == 'real':
                printer.print("double {}[{}];".format(p.prop_name, len(self.setup)))
            elif p.prop_type == 'vector':
                printer.print("double {}[{}][3];".format(p.prop_name, len(self.setup)))
            else:
                raise Exception("Invalid property type!")

    def generate_setup(self):
        index = 0
        for ps in self.setup:
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

    def generate_volatile_reset(self):
        printer.print("for(int i = 0; i < {}; i++) {{".format(len(self.setup)))
        printer.add_ind(4)

        for p in self.properties:
            if p.volatile is True:
                if p.prop_type == 'vector':
                    printer.print('{}[i][0] = 0.0;'.format(p.prop_name))
                    printer.print('{}[i][1] = 0.0;'.format(p.prop_name))
                    printer.print('{}[i][2] = 0.0;'.format(p.prop_name))

        printer.add_ind(-4)
        printer.print("}")

    def generate(self):
        printer.print("int main() {")
        printer.add_ind(4)
        printer.print("const int nparticles = {};".format(len(self.setup)))
        self.generate_properties_decl()
        self.generate_setup()
        printer.print("for(int t = 0; t < {}; t++) {{".format(self.ntimesteps))
        printer.add_ind(4)
        self.generate_volatile_reset()
        for block in self.blocks:
            block.generate()
        printer.add_ind(-4)
        printer.print("}")
        printer.add_ind(-4)
        printer.print("}")

