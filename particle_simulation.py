from assign import AssignAST
from block import BlockAST
from branches import BranchAST
from data_types import Type_Int, Type_Float, Type_Vector
from expr import ExprAST
from loops import ForAST, ParticleForAST, NeighborForAST
from properties import Property
from printer import printer

class ParticleSimulation:
    def __init__(self):
        self.properties = []
        self.defaults = {}
        self.setup = []
        self.grid_config = []
        self.setup_blocks = []
        self.blocks = []
        self.produced_stmts = []
        self.ntimesteps = 0
        self.expr_id = 0
        self.iter_id = 0

    def add_property(self, prop_name, prop_type, value, volatile):
        prop = Property(self, prop_name, prop_type, value, volatile)
        self.properties.append(prop)
        self.defaults[prop_name] = value
        return prop

    def add_real_property(self, prop_name, value=0.0, volatile=False):
        return self.add_property(prop_name, Type_Float, value, volatile)

    def add_vector_property(self, prop_name, value=[0.0, 0.0, 0.0], volatile=False):
        return self.add_property(prop_name, Type_Vector, value, volatile)

    def new_expr(self):
        self.expr_id += 1
        return self.expr_id - 1

    def new_iter(self):
        self.iter_id += 1
        return self.iter_id - 1

    def setup_grid(self, config):
        self.grid_config = config

    def create_particle_lattice(self, config, spacing, props={}):
        nx = int((config[0][1] - config[0][0]) / spacing[0] - 0.001) + 1 
        ny = int((config[1][1] - config[1][0]) / spacing[1] - 0.001) + 1 
        nz = int((config[2][1] - config[2][0]) / spacing[2] - 0.001) + 1 

        xi = ForAST(self, 0, nx)
        yi = ForAST(self, 0, ny)
        zi = ForAST(self, 0, nz)

        particle_props = self.defaults.copy()
        for p in props:
            particle_props[p] = props[p]

        positions = [p for p in self.properties if p.name() == 'position'][0]
        index = (xi.iter() * ny + yi.iter()) * nz + zi.iter()
        pos = [ 
            config[0][0] + spacing[0] * xi.iter(),
            config[1][0] + spacing[1] * yi.iter(),
            config[2][0] + spacing[2] * zi.iter() 
        ]

        xi.set_body(yi)
        yi.set_body(zi)
        zi.set_body(BlockAST([AssignAST(positions[index], pos)]))
        self.setup_blocks.append(BlockAST([xi]))

    def setup_cell_lists(self, cutoff_radius):
        ncells = [ 
            (self.grid_config[0][1] - self.grid_config[0][0]) / cutoff_radius,
            (self.grid_config[1][1] - self.grid_config[1][0]) / cutoff_radius,
            (self.grid_config[2][1] - self.grid_config[2][0]) / cutoff_radius
        ]   

    def set_timesteps(self, ts):
        self.ntimesteps = ts

    def particle_pairs(self, cutoff_radius=None, position=None):
        i = ParticleForAST(self)
        j = NeighborForAST(self, i.iter())
        i.set_body(j)

        if cutoff_radius is not None and position is not None:
            delta = position[i.iter()] - position[j.iter()]
            rsq = self.vector_len_sq(delta)
            yield i.iter(), j.iter(), delta, rsq
            j.set_body(BranchAST(rsq < cutoff_radius, self.produced_stmts.copy(), None))

        else:
            yield i.iter(), j.iter()
            j.set_body(self.produced_stmts.copy())

        self.blocks.append(BlockAST([i]))
        #self.produced_stmts = []

    def particles(self):
        i = ParticleForAST(self)
        yield i.iter()
        i.set_body(BlockAST(self.produced_stmts.copy()))
        self.blocks.append(BlockAST([i]))
        #self.produced_stmts = []

    def vector_len_sq(self, expr):
        return ExprAST(self, expr, None, 'vector_len_sq')

    def generate_properties_decl(self):
        for p in self.properties:
            if p.prop_type == Type_Float:
                printer.print("double {}[{}];".format(p.prop_name, len(self.setup)))
            elif p.prop_type == Type_Vector:
                printer.print("double {}[{}][3];".format(p.prop_name, len(self.setup)))
            else:
                raise Exception("Invalid property type!")

    def generate_setup(self):
        for block in self.setup_blocks:
            block.generate()

    def generate_volatile_reset(self):
        printer.print("for(int i = 0; i < {}; i++) {{".format(len(self.setup)))
        printer.add_ind(4)

        for p in self.properties:
            if p.volatile is True:
                if p.prop_type == Type_Vector:
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

