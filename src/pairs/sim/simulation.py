from pairs.ir.arrays import Arrays
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.layouts import Layouts
from pairs.ir.module import Module
from pairs.ir.properties import Properties
from pairs.ir.symbols import Symbol
from pairs.ir.types import Types
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
from pairs.sim.timestep import Timestep
from pairs.sim.variables import VariablesDecl
from pairs.sim.vtk import VTKWrite
from pairs.transformations import Transformations


class Simulation:
    def __init__(self, code_gen, dims=3, timesteps=100, particle_capacity=10000):
        self.code_gen = code_gen
        self.code_gen.assign_simulation(self)
        self.position_prop = None
        self.properties = Properties(self)
        self.vars = Variables(self)
        self.arrays = Arrays(self)
        self.particle_capacity = self.add_var('particle_capacity', Types.Int32, particle_capacity)
        self.nlocal = self.add_var('nlocal', Types.Int32)
        self.nghost = self.add_var('nghost', Types.Int32)
        self.resizes = self.add_array('resizes', 3, Types.Int32)
        self.grid = None
        self.cell_lists = None
        self.neighbor_lists = None
        self.pbc = None
        self.scope = []
        self.nested_count = 0
        self.nest = False
        self.check_decl_usage = True
        self._block = Block(self, [])
        self.setups = Block(self, [])
        self.kernels = Block(self, [])
        self.module_list = []
        self._check_properties_resize = False
        self._resizes_to_check = {}
        self._module_name = None
        self.dims = dims
        self.ntimesteps = timesteps
        self.expr_id = 0
        self.iter_id = 0
        self.vtk_file = None
        self._target = None
        self.nparticles = self.nlocal + self.nghost
        self.properties.add_capacity(self.particle_capacity)

    def add_module(self, module):
        assert isinstance(module, Module), "add_module(): Given parameter is not of type Module!"
        self.module_list.append(module)

    def modules(self):
        sorted_mods = []
        main_mod = None
        for m in self.module_list:
            if m.name != 'main':
                sorted_mods.append(m)
            else:
                main_mod = m

        return sorted_mods + [main_mod]

    def ndims(self):
        return self.dims

    def add_real_property(self, prop_name, value=0.0, vol=False):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        return self.properties.add(prop_name, Types.Double, value, vol)

    def add_position(self, prop_name, value=[0.0, 0.0, 0.0], vol=False, layout=Layouts.AoS):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        self.position_prop = self.properties.add(prop_name, Types.Vector, value, vol, layout)
        return self.position_prop

    def add_vector_property(self, prop_name, value=[0.0, 0.0, 0.0], vol=False, layout=Layouts.AoS):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        return self.properties.add(prop_name, Types.Vector, value, vol, layout)

    def property(self, prop_name):
        return self.properties.find(prop_name)

    def position(self):
        return self.position_prop

    def add_array(self, arr_name, arr_sizes, arr_type, arr_layout=Layouts.AoS):
        assert self.array(arr_name) is None, f"Array already defined: {arr_name}"
        return self.arrays.add(arr_name, arr_sizes, arr_type, arr_layout)

    def add_static_array(self, arr_name, arr_sizes, arr_type, arr_layout=Layouts.AoS, init_value=None):
        assert self.array(arr_name) is None, f"Array already defined: {arr_name}"
        return self.arrays.add_static(arr_name, arr_sizes, arr_type, arr_layout, init_value=init_value)

    def array(self, arr_name):
        return self.arrays.find(arr_name)

    def add_var(self, var_name, var_type, init_value=0):
        assert self.var(var_name) is None, f"Variable already defined: {var_name}"
        return self.vars.add(var_name, var_type, init_value)

    def add_symbol(self, sym_type):
        return Symbol(self, sym_type)

    def var(self, var_name):
        return self.vars.find(var_name)

    def grid_2d(self, xmin, xmax, ymin, ymax):
        self.grid = Grid2D(self, xmin, xmax, ymin, ymax)
        return self.grid

    def grid_3d(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.grid = Grid3D(self, xmin, xmax, ymin, ymax, zmin, zmax)
        return self.grid

    def create_particle_lattice(self, grid, spacing, props={}):
        positions = self.position()
        lattice = ParticleLattice(self, grid, spacing, props, positions)
        self.setups.add_statement(lattice)

    def from_file(self, filename, prop_names):
        props = [self.property(prop_name) for prop_name in prop_names]
        read_object = ReadFromFile(self, filename, props)
        self.setups.add_statement(read_object)
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

    def compute(self, func, cutoff_radius=None, symbols={}):
        return compute(self, func, cutoff_radius, symbols)

    def init_block(self):
        self._block = Block(self, [])
        self._check_properties_resize = False
        self._resizes_to_check = {}
        self._module_name = None

    def module_name(self, name):
        self._module_name = name

    def check_properties_resize(self):
        self._check_properties_resize = True

    def check_resize(self, capacity, size):
        if capacity not in self._resizes_to_check:
            self._resizes_to_check[capacity] = size
        else:
            raise Exception("Two sizes assigned to same capacity!")

    def build_kernel_block_with_statements(self):
        self.kernels.add_statement(
            Module(self,
                name=self._module_name,
                block=Block(self, self._block),
                resizes_to_check=self._resizes_to_check,
                check_properties_resize=self._check_properties_resize,
                run_on_device=True))

    def add_statement(self, stmt):
        if not self.scope:
            self._block.add_statement(stmt)
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

    def enter(self, scope):
        self.scope.append(scope)

    def leave(self):
        if not self.nest:
            self.scope.pop()
        else:
            self.nested_count += 1

    def vtk_output(self, filename):
        self.vtk_file = filename

    def target(self, target):
        self._target = target

    def generate(self):
        assert self._target is not None, "Target not specified!"

        timestep = Timestep(self, self.ntimesteps, [
            (EnforcePBC(self, self.pbc), 20),
            (SetupPBC(self, self.pbc), UpdatePBC(self, self.pbc), 20),
            (CellListsBuild(self, self.cell_lists), 20),
            (NeighborListsBuild(self, self.neighbor_lists), 20),
            PropertiesResetVolatile(self),
            self.kernels
        ])

        self.enter(timestep.block)
        timestep.add(VTKWrite(self, self.vtk_file, timestep.timestep() + 1))
        self.leave()

        body = Block.from_list(self, [
            self.setups,
            CellListsStencilBuild(self, self.cell_lists),
            VTKWrite(self, self.vtk_file, 0),
            timestep.as_block()
        ])

        decls = Block.from_list(self, [
            VariablesDecl(self),
            ArraysDecl(self),
            PropertiesAlloc(self),
        ])

        program = Module(self, name='main', block=Block.merge_blocks(decls, body))

        # Apply transformations
        transformations = Transformations(program, self._target)
        transformations.apply_all()

        # For this part on, all bin ops are generated without usage verification
        self.check_decl_usage = False

        ASTGraph(self.kernels, "kernels").render()
        self.code_gen.generate_program(program)
