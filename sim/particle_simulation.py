from ast.arrays import Arrays
from ast.block import BlockAST
from ast.branches import BranchAST
from ast.data_types import Type_Int, Type_Float, Type_Vector
from ast.loops import ParticleForAST, NeighborForAST
from ast.properties import Properties
from ast.transform import Transform
from ast.variables import Variables
from sim.cell_lists import CellLists, CellListsBuild, CellListsStencilBuild
from sim.lattice import ParticleLattice
from sim.properties import PropertiesDecl, PropertiesResetVolatile
from sim.timestep import Timestep


class ParticleSimulation:
    def __init__(self, code_gen, dims=3, timesteps=100):
        self.code_gen = code_gen
        self.properties = Properties(self)
        self.vars = Variables(self)
        self.arrays = Arrays(self)
        self.nparticles = self.add_var('nparticles', Type_Int)
        self.setup_blocks = []
        self.grid_config = []
        self.captured_stmts = []
        self.capture_buffer = []
        self.capture = False
        self.dimensions = dims
        self.ntimesteps = timesteps
        self.expr_id = 0
        self.iter_id = 0
        self.cell_lists = CellLists(self, 2.8, 2.8)

    def add_real_property(self, prop_name, value=0.0, vol=False):
        return self.properties.add(prop_name, Type_Float, value, vol)

    def add_vector_property(self, prop_name, value=[0.0, 0.0, 0.0], vol=False):
        return self.properties.add(prop_name, Type_Vector, value, vol)

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
        block = ParticleLattice(self, config, spacing, props, positions)
        self.setup_blocks.append(block.lower())

    def particle_pairs(self, cutoff_radius=None, position=None):
        i = ParticleForAST(self)
        j = NeighborForAST(self, i.iter())
        i.set_body(BlockAST(self, [j]))

        if cutoff_radius is not None and position is not None:
            dp = position[i.iter()] - position[j.iter()]
            rsq = dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]
            self.start_capture()
            yield i.iter(), j.iter(), dp, rsq
            self.stop_capture()
            j.set_body(BlockAST(self, [
                BranchAST(self, rsq < cutoff_radius,
                          BlockAST(self, self.capture_buffer.copy()), None)]))

        else:
            yield i.iter(), j.iter()
            j.set_body(BlockAST(self, self.capture_buffer.copy()))

        self.captured_stmts.append(i)

    def particles(self):
        i = ParticleForAST(self)
        self.start_capture()
        yield i.iter()
        self.stop_capture()
        i.set_body(BlockAST(self, self.capture_buffer.copy()))
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

    def generate(self):
        program = BlockAST.from_list(self, [
            PropertiesDecl(self).lower(),
            CellListsStencilBuild(self, self.cell_lists).lower(),
            BlockAST.from_list(self, self.setup_blocks),
            Timestep(self, self.ntimesteps, [
                (CellListsBuild(self, self.cell_lists).lower(), 20),
                PropertiesResetVolatile(self).lower(),
                self.captured_stmts
            ]).as_block()
        ])

        program.transform(Transform.flatten)
        program.transform(Transform.simplify)

        self.code_gen.generate_program_preamble()
        program.generate()
        self.code_gen.generate_program_epilogue()
