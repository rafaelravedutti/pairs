from arrays import Arrays
from assign import AssignAST
from block import BlockAST
from branches import BranchAST
from cell_lists import CellLists
from data_types import Type_Int, Type_Float, Type_Vector
from expr import ExprAST
from lattice import ParticleLattice
from loops import ForAST, ParticleForAST, NeighborForAST
from properties import Properties
from printer import printer
from timestep import Timestep
from transform import Transform
from variables import Variables

class ParticleSimulation:
    def __init__(self, dims=3, timesteps=100):
        self.properties = Properties(self)
        self.vars = Variables(self)
        self.arrays = Arrays(self)
        self.setup_blocks = []
        self.grid_config = []
        self.captured_stmts = []
        self.capture_buffer = []
        self.capture = False
        self.dimensions = dims
        self.ntimesteps = timesteps
        self.expr_id = 0
        self.iter_id = 0
        self.nparticles = 0

    def add_real_property(self, prop_name, value=0.0, volatile=False):
        return self.properties.add(prop_name, Type_Float, value, volatile)

    def add_vector_property(self, prop_name, value=[0.0, 0.0, 0.0], volatile=False):
        return self.properties.add(prop_name, Type_Vector, value, volatile)

    def property(self, prop_name):
        return self.properties.find(prop_name)

    def add_array(self, array_name, array_sizes, array_type):
        return self.arrays.add(array_name, array_sizes, array_type)

    def array(self, array_name):
        return self.arrays.find(array_name)

    def add_var(self, var_name, var_type):
        return self.vars.add(var_name, var_type)

    def var(self, var_name):
        return self.vars.find(var_name)

    def new_expr(self):
        self.expr_id += 1
        return self.expr_id - 1

    def new_iter(self):
        self.iter_id += 1
        return self.iter_id - 1

    def setup_grid(self, config):
        self.grid_config = config

    def create_particle_lattice(self, config, spacing, props={}):
        positions = self.property('position')
        block, nparticles = ParticleLattice(self, config, spacing, props, positions).lower()
        self.setup_blocks.append(block)
        self.nparticles += nparticles

    def particle_pairs(self, cutoff_radius=None, position=None):
        i = ParticleForAST(self)
        j = NeighborForAST(self, i.iter())
        i.set_body(BlockAST([j]))

        if cutoff_radius is not None and position is not None:
            delta = position[i.iter()] - position[j.iter()]
            rsq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
            self.start_capture()
            yield i.iter(), j.iter(), delta, rsq
            self.stop_capture()
            j.set_body(BlockAST([BranchAST(rsq < cutoff_radius, BlockAST(self.capture_buffer.copy()), None)]))

        else:
            yield i.iter(), j.iter()
            j.set_body(BlockAST(self.capture_buffer.copy()))

        self.captured_stmts.append(i)

    def particles(self):
        i = ParticleForAST(self)
        self.start_capture()
        yield i.iter()
        self.stop_capture()
        i.set_body(BlockAST(self.capture_buffer.copy()))
        self.captured_stmts.append(i)

    def start_capture(self):
        self.capture_buffer = []
        self.capture = True

    def stop_capture(self):
        self.capture = False

    def capture_statement(self, stmt):
        if self.capture is True:
            self.capture_buffer.append(stmt)

        return stmt

    def generate_properties_decl(self):
        for p in self.properties.all():
            if p.prop_type == Type_Float:
                printer.print(f"    double {p.prop_name}[{self.nparticles}];")
            elif p.prop_type == Type_Vector:
                if p.flattened:
                    printer.print(f"    double {p.prop_name}[{self.nparticles * self.dimensions}];")
                else:
                    printer.print(f"    double {p.prop_name}[{self.nparticles}][{self.dimensions}];")
            else:
                raise Exception("Invalid property type!")

    def generate(self):
        printer.print("int main() {")
        printer.print(f"    const int nparticles = {self.nparticles};")
        reset_loop = ParticleForAST(self)
        reset_loop.set_body(BlockAST([AssignAST(self, p[reset_loop.iter()], 0.0) for p in self.properties.volatiles()]))
        cell_lists = CellLists(self, 2.8, 2.8)
        timestep_loop = Timestep(self, self.ntimesteps)
        timestep_loop.add(cell_lists.build(), 20)
        timestep_loop.add(reset_loop)
        timestep_loop.add(self.captured_stmts)
        program = BlockAST.merge_blocks(BlockAST.from_list(self.setup_blocks), timestep_loop.as_block())
        program.transform(Transform.flatten)
        program.transform(Transform.simplify)
        self.generate_properties_decl()
        program.generate()
        printer.print("}")
