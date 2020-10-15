from ast.arrays import Arrays
from ast.block import Block
from ast.branches import Filter
from ast.data_types import Type_Int, Type_Float, Type_Vector
from ast.layouts import Layout_AoS
from ast.loops import ParticleFor, NeighborFor
from ast.properties import Properties
from ast.transform import Transform
from ast.variables import Variables
from sim.arrays import ArraysDecl
from sim.cell_lists import CellLists, CellListsBuild, CellListsStencilBuild
from sim.kernel_wrapper import KernelWrapper
from sim.lattice import ParticleLattice
from sim.properties import PropertiesDecl, PropertiesResetVolatile
from sim.setup_wrapper import SetupWrapper
from sim.timestep import Timestep
from sim.variables import VariablesDecl


class ParticleSimulation:
    def __init__(self, code_gen, dims=3, timesteps=100):
        self.code_gen = code_gen
        self.global_scope = None
        self.properties = Properties(self)
        self.vars = Variables(self)
        self.arrays = Arrays(self)
        self.nparticles = self.add_var('nparticles', Type_Int)
        self.grid_config = []
        self.scope = []
        self.nested_count = 0
        self.nest = False
        self.block = Block(self, [])
        self.setups = SetupWrapper()
        self.kernels = KernelWrapper()
        self.dimensions = dims
        self.ntimesteps = timesteps
        self.expr_id = 0
        self.iter_id = 0
        self.cell_lists = CellLists(self, 2.8, 2.8)

    def add_real_property(self, prop_name, value=0.0, vol=False):
        return self.properties.add(prop_name, Type_Float, value, vol)

    def add_vector_property(
            self, prop_name, value=[0.0, 0.0, 0.0],
            vol=False, layout=Layout_AoS):
        return self.properties.add(prop_name, Type_Vector, value, vol, layout)

    def property(self, prop_name):
        return self.properties.find(prop_name)

    def add_array(self, arr_name, arr_sizes, arr_type, arr_layout=Layout_AoS):
        return self.arrays.add(arr_name, arr_sizes, arr_type, arr_layout)

    def array(self, arr_name):
        return self.arrays.find(arr_name)

    def add_var(self, var_name, var_type, init_value=0):
        return self.vars.add(var_name, var_type, init_value)

    def var(self, var_name):
        return self.vars.find(var_name)

    def setup_grid(self, config):
        self.grid_config = config

    def create_particle_lattice(self, config, spacing, props={}):
        positions = self.property('position')
        lattice = ParticleLattice(self, config, spacing, props, positions)
        self.setups.add_setup_block(lattice.lower())

    def particle_pairs(self, cutoff_radius=None, position=None):
        self.clear_block()
        for i in ParticleFor(self):
            for j in NeighborFor(self, i, self.cell_lists):
                if cutoff_radius is not None and position is not None:
                    dp = position[i] - position[j]
                    rsq = dp.x() * dp.x() + dp.y() * dp.y() + dp.z() * dp.z()
                    for _ in Filter(self, rsq < cutoff_radius):
                        yield i, j, dp, rsq

                else:
                    yield i, j

        self.kernels.add_kernel_block(self.block)

    def particles(self):
        self.clear_block()
        for i in ParticleFor(self):
            yield i

        self.kernels.add_kernel_block(self.block)

    def clear_block(self):
        self.block = Block(self, [])

    def add_statement(self, stmt):
        if not self.scope:
            self.block.add_statement(stmt)
        else:
            self.scope[-1].add_statement(stmt)

        return stmt

    def nest_mode(self):
        self.nested_count = 0
        self.nest = True
        yield
        self.nest = False
        for _ in range(0, self.nested_count):
            self.scope.pop()

    def enter_scope(self, scope):
        self.scope.append(scope)

    def leave_scope(self):
        if not self.nest:
            self.scope.pop()
        else:
            self.nested_count += 1

    def generate(self):
        program = Block.from_list(self, [
            VariablesDecl(self).lower(),
            ArraysDecl(self).lower(),
            PropertiesDecl(self).lower(),
            CellListsStencilBuild(self, self.cell_lists).lower(),
            self.setups.lower(),
            Timestep(self, self.ntimesteps, [
                (CellListsBuild(self, self.cell_lists).lower(), 20),
                PropertiesResetVolatile(self).lower(),
                self.kernels.lower()
            ]).as_block()
        ])

        self.global_scope = program
        Block.set_block_levels(program)
        Transform.apply(program, Transform.flatten)
        Transform.apply(program, Transform.simplify)
        Transform.apply(program, Transform.reuse_index_expressions)
        Transform.apply(program, Transform.reuse_expr_expressions)
        Transform.apply(program, Transform.reuse_array_access_expressions)
        Transform.apply(program, Transform.move_loop_invariant_expressions)

        self.code_gen.generate_program_preamble()
        program.generate()
        self.code_gen.generate_program_epilogue()
