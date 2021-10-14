from pairs.ir.arrays import Arrays
from pairs.ir.block import Block, KernelBlock
from pairs.ir.branches import Filter
from pairs.ir.data_types import Type_Int, Type_Float, Type_Vector
from pairs.ir.layouts import Layout_AoS
from pairs.ir.loops import ParticleFor, NeighborFor
from pairs.ir.properties import Properties
from pairs.ir.variables import Variables
from pairs.graph.graphviz import ASTGraph
from pairs.mapping.funcs import compute
from pairs.sim.arrays import ArraysDecl
from pairs.sim.cell_lists import CellLists, CellListsBuild, CellListsStencilBuild
from pairs.sim.grid import Grid2D, Grid3D
from pairs.sim.lattice import ParticleLattice
from pairs.sim.neighbor_lists import NeighborLists, NeighborListsBuild
from pairs.sim.pbc import PBC, UpdatePBC, EnforcePBC, SetupPBC
from pairs.sim.properties import PropertiesAlloc, PropertiesResetVolatile
from pairs.sim.read_from_file import ReadFromFile
from pairs.sim.setup_wrapper import SetupWrapper
from pairs.sim.timestep import Timestep
from pairs.sim.variables import VariablesDecl
from pairs.sim.vtk import VTKWrite
from pairs.transformations.add_device_copies import add_device_copies
from pairs.transformations.prioritize_scalar_ops import prioritaze_scalar_ops
from pairs.transformations.set_used_bin_ops import set_used_bin_ops
from pairs.transformations.simplify import simplify_expressions
from pairs.transformations.LICM import move_loop_invariant_code


class ParticleSimulation:
    def __init__(self, code_gen, dims=3, timesteps=100):
        self.code_gen = code_gen
        self.code_gen.assign_simulation(self)
        self.global_scope = None
        self.properties = Properties(self)
        self.vars = Variables(self)
        self.arrays = Arrays(self)
        self.particle_capacity = self.add_var('particle_capacity', Type_Int, 10000)
        self.nlocal = self.add_var('nlocal', Type_Int)
        self.nghost = self.add_var('nghost', Type_Int)
        self.grid = None
        self.cell_lists = None
        self.neighbor_lists = None
        self.pbc = None
        self.scope = []
        self.nested_count = 0
        self.nest = False
        self.check_decl_usage = True
        self.block = Block(self, [])
        self.setups = SetupWrapper(self)
        self.kernels = Block(self, [])
        self.dims = dims
        self.ntimesteps = timesteps
        self.expr_id = 0
        self.iter_id = 0
        self.vtk_file = None
        self.nparticles = self.nlocal + self.nghost
        self.properties.add_capacity(self.particle_capacity)

    def ndims(self):
        return self.dims

    def add_real_property(self, prop_name, value=0.0, vol=False):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        return self.properties.add(prop_name, Type_Float, value, vol)

    def add_vector_property(self, prop_name, value=[0.0, 0.0, 0.0], vol=False, layout=Layout_AoS):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        return self.properties.add(prop_name, Type_Vector, value, vol, layout)

    def property(self, prop_name):
        return self.properties.find(prop_name)

    def add_array(self, arr_name, arr_sizes, arr_type, arr_layout=Layout_AoS):
        assert self.array(arr_name) is None, f"Array already defined: {arr_name}"
        return self.arrays.add(arr_name, arr_sizes, arr_type, arr_layout)

    def add_static_array(self, arr_name, arr_sizes, arr_type, arr_layout=Layout_AoS):
        assert self.array(arr_name) is None, f"Array already defined: {arr_name}"
        return self.arrays.add_static(arr_name, arr_sizes, arr_type, arr_layout)

    def array(self, arr_name):
        return self.arrays.find(arr_name)

    def add_var(self, var_name, var_type, init_value=0):
        assert self.var(var_name) is None, f"Variable already defined: {var_name}"
        return self.vars.add(var_name, var_type, init_value)

    def add_or_reuse_var(self, var_name, var_type, init_value=0):
        existing_var = self.var(var_name)
        if existing_var is not None:
            assert existing_var.type() == var_type, f"Cannot reuse variable {var_name}: types differ!"
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

    def from_file(self, filename, prop_names):
        props = [self.property(prop_name) for prop_name in prop_names]
        read_object = ReadFromFile(self, filename, props)
        self.setups.add_setup_block(read_object.lower())
        self.grid = read_object.grid

    def create_cell_lists(self, spacing, cutoff_radius):
        self.cell_lists = CellLists(self, self.grid, spacing, cutoff_radius)
        return self.cell_lists

    def create_neighbor_lists(self):
        self.neighbor_lists = NeighborLists(self.cell_lists)
        return self.neighbor_lists

    def periodic(self, cutneigh, flags=[1, 1, 1]):
        self.pbc = PBC(self, self.grid, cutneigh, flags)
        self.properties.add_capacity(self.pbc.pbc_capacity)
        return self.pbc

    def compute(self, sim, func, cutoff_radius=None, position=None, symbols={}):
        return compute(sim, func, cutoff_radius, position, symbols)

    def particle_pairs(self, cutoff_radius=None, position=None):
        self.clear_block()
        for i in ParticleFor(self):
            for j in NeighborFor(self, i, self.cell_lists, self.neighbor_lists):
                if cutoff_radius is not None and position is not None:
                    dp = position[i] - position[j]
                    rsq = dp.x() * dp.x() + dp.y() * dp.y() + dp.z() * dp.z()
                    for _ in Filter(self, rsq < cutoff_radius):
                        yield i, j, dp, rsq

                else:
                    yield i, j

        self.kernels.add_statement(KernelBlock(self, self.block))

    def particles(self):
        self.clear_block()
        for i in ParticleFor(self):
            yield i

        self.kernels.add_statement(KernelBlock(self, self.block))

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

    def vtk_output(self, filename):
        self.vtk_file = filename

    def generate(self):
        timestep = Timestep(self, self.ntimesteps, [
            (EnforcePBC(self.pbc).lower(), 20),
            (SetupPBC(self.pbc).lower(), UpdatePBC(self.pbc).lower(), 20),
            (CellListsBuild(self.cell_lists).lower(), 20),
            (NeighborListsBuild(self.neighbor_lists).lower(), 20),
            PropertiesResetVolatile(self).lower(),
            self.kernels
        ])

        timestep.add(VTKWrite(self, self.vtk_file, timestep.timestep() + 1).lower())

        body = Block.from_list(self, [
            self.setups.lower(),
            CellListsStencilBuild(self.cell_lists).lower(),
            VTKWrite(self, self.vtk_file, 0).lower(),
            timestep.as_block()
        ])

        decls = Block.from_list(self, [
            VariablesDecl(self).lower(),
            ArraysDecl(self).lower(),
            PropertiesAlloc(self).lower(),
        ])

        program = Block.merge_blocks(decls, body)
        self.global_scope = program

        # Transformations
        prioritaze_scalar_ops(program)
        simplify_expressions(program)
        move_loop_invariant_code(program)
        set_used_bin_ops(program)
        add_device_copies(program)

        # For this part on, all bin ops are generated without usage verification
        self.check_decl_usage = False

        ASTGraph(self.kernels, "kernels").render()
        self.code_gen.generate_program(program)
