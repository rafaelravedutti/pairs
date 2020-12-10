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
from sim.grid import Grid2D, Grid3D
from sim.kernel_wrapper import KernelWrapper
from sim.lattice import ParticleLattice
from sim.pbc import PBC, UpdatePBC, EnforcePBC, SetupPBC
from sim.properties import PropertiesAlloc, PropertiesResetVolatile
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
        self.particle_capacity = self.add_var('particle_capacity', Type_Int, 10000)
        self.nlocal = self.add_var('nlocal', Type_Int)
        self.nghost = self.add_var('nghost', Type_Int)
        self.nparticles = self.nlocal + self.nghost
        self.grid = None
        self.cell_lists = None
        self.pbc = None
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
        self.properties.add_capacity(self.particle_capacity)

    def add_real_property(self, prop_name, value=0.0, vol=False):
        assert self.property(prop_name) is None, \
            f"add_real_property(): Property already defined: {prop_name}"
        return self.properties.add(prop_name, Type_Float, value, vol)

    def add_vector_property(self, prop_name, value=[0.0, 0.0, 0.0], vol=False, layout=Layout_AoS):
        assert self.property(prop_name) is None, \
            f"add_vector_property(): Property already defined: {prop_name}"
        return self.properties.add(prop_name, Type_Vector, value, vol, layout)

    def property(self, prop_name):
        return self.properties.find(prop_name)

    def add_array(self, arr_name, arr_sizes, arr_type, arr_layout=Layout_AoS):
        assert self.array(arr_name) is None, \
            f"add_array(): Array already defined: {arr_name}"
        return self.arrays.add(arr_name, arr_sizes, arr_type, arr_layout)

    def add_static_array(self, arr_name, arr_sizes, arr_type, arr_layout=Layout_AoS):
        assert self.array(arr_name) is None, \
            f"add_static_array(): Array already defined: {arr_name}"
        return self.arrays.add_static(arr_name, arr_sizes, arr_type, arr_layout)

    def array(self, arr_name):
        return self.arrays.find(arr_name)

    def add_var(self, var_name, var_type, init_value=0):
        assert self.var(var_name) is None, \
            f"add_var(): Variable already defined: {var_name}"
        return self.vars.add(var_name, var_type, init_value)

    def add_or_reuse_var(self, var_name, var_type, init_value=0):
        existing_var = self.var(var_name)
        if existing_var is not None:
            assert existing_var.type() == var_type, \
                f"add_or_reuse_var(): Cannot reuse variable {var_name}: types differ!"
            return existing_var

        return self.vars.add(var_name, var_type, init_value)

    def var(self, var_name):
        return self.vars.find(var_name)

    def grid_2d(self, xmin, xmax, ymin, ymax):
        self.grid = Grid2D(self, xmin, xmax, ymin, ymax)
        return self.grid

    def grid_3d(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.grid = Grid3D(self, xmin, xmax, ymin, ymax, zmin, zmax)
        return self.grid

    def create_particle_lattice(self, grid, spacing, props={}):
        positions = self.property('position')
        lattice = ParticleLattice(self, grid, spacing, props, positions)
        self.setups.add_setup_block(lattice.lower())

    def create_cell_lists(self, grid, spacing, cutoff_radius):
        self.cell_lists = CellLists(self, grid, spacing, cutoff_radius)
        return self.cell_lists

    def periodic(self, cutneigh, flags=[1, 1, 1]):
        self.pbc = PBC(self, self.grid, cutneigh, flags)
        self.properties.add_capacity(self.pbc.pbc_capacity)
        return self.pbc

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
        body = Block.from_list(self, [
            CellListsStencilBuild(self.cell_lists).lower(),
            self.setups.lower(),
            Timestep(self, self.ntimesteps, [
                (EnforcePBC(self.pbc).lower(), 20),
                (SetupPBC(self.pbc).lower(), 20),
                UpdatePBC(self.pbc).lower(),
                (CellListsBuild(self.cell_lists).lower(), 20),
                PropertiesResetVolatile(self).lower(),
                self.kernels.lower()
            ]).as_block()
        ])

        decls = Block.from_list(self, [
            VariablesDecl(self).lower(),
            ArraysDecl(self).lower(),
            PropertiesAlloc(self).lower(),
        ])

        program = Block.merge_blocks(decls, body)
        self.global_scope = program
        Block.set_block_levels(program)
        Transform.apply(program, Transform.flatten)
        Transform.apply(program, Transform.simplify)
        #Transform.apply(program, Transform.reuse_index_expressions)
        #Transform.apply(program, Transform.reuse_expr_expressions)
        #Transform.apply(program, Transform.reuse_array_access_expressions)
        #Transform.apply(program, Transform.move_loop_invariant_expressions)

        self.code_gen.generate_program_preamble()
        program.generate()
        self.code_gen.generate_program_epilogue()
