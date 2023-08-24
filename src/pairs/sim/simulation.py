from pairs.ir.arrays import Arrays
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.features import Features, FeatureProperties
from pairs.ir.functions import Call_Void
from pairs.ir.kernel import Kernel
from pairs.ir.layouts import Layouts
from pairs.ir.module import Module
from pairs.ir.properties import Properties, ContactProperties
from pairs.ir.symbols import Symbol
from pairs.ir.types import Types
from pairs.ir.variables import Variables
from pairs.graph.graphviz import ASTGraph
from pairs.mapping.funcs import compute
from pairs.sim.arrays import ArraysDecl
from pairs.sim.cell_lists import CellLists, CellListsBuild, CellListsStencilBuild
from pairs.sim.comm import Comm
from pairs.sim.contact_history import ContactHistory, BuildContactHistory
from pairs.sim.domain_partitioning import DimensionRanges
from pairs.sim.features import FeaturePropertiesAlloc 
from pairs.sim.grid import Grid2D, Grid3D
from pairs.sim.lattice import ParticleLattice
from pairs.sim.neighbor_lists import NeighborLists, NeighborListsBuild
from pairs.sim.pbc import EnforcePBC
from pairs.sim.properties import ContactPropertiesAlloc, PropertiesAlloc, PropertiesResetVolatile
from pairs.sim.read_from_file import ReadParticleData
from pairs.sim.timestep import Timestep
from pairs.sim.variables import VariablesDecl
from pairs.sim.vtk import VTKWrite
from pairs.transformations import Transformations


class Simulation:
    def __init__(self, code_gen, dims=3, timesteps=100):
        self.code_gen = code_gen
        self.code_gen.assign_simulation(self)
        self.position_prop = None
        self.properties = Properties(self)
        self.vars = Variables(self)
        self.arrays = Arrays(self)
        self.features = Features(self)
        self.feature_properties = FeatureProperties(self)
        self.contact_properties = ContactProperties(self)
        self.particle_capacity = self.add_var('particle_capacity', Types.Int32, 200000)
        self.neighbor_capacity = self.add_var('neighbor_capacity', Types.Int32, 100)
        self.nlocal = self.add_var('nlocal', Types.Int32)
        self.nghost = self.add_var('nghost', Types.Int32)
        self.resizes = self.add_array('resizes', 3, Types.Int32, arr_sync=False)
        self.grid = None
        self.cell_lists = None
        self.neighbor_lists = None
        self.scope = []
        self.nested_count = 0
        self.nest = False
        self._capture_statements = True
        self._block = Block(self, [])
        self.setups = Block(self, [])
        self.functions = Block(self, [])
        self.module_list = []
        self.kernel_list = []
        self._check_properties_resize = False
        self._resizes_to_check = {}
        self._module_name = None
        self.dims = dims
        self.ntimesteps = timesteps
        self.expr_id = 0
        self.iter_id = 0
        self.vtk_file = None
        self._target = None
        self._dom_part = DimensionRanges(self)
        self.nparticles = self.nlocal + self.nghost

    def add_module(self, module):
        assert isinstance(module, Module), "add_module(): Given parameter is not of type Module!"
        if module.name not in [m.name for m in self.module_list]:
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

    def add_kernel(self, kernel):
        assert isinstance(kernel, Kernel), "add_kernel(): Given parameter is not of type Kernel!"
        self.kernel_list.append(kernel)

    def kernels(self):
        return self.kernel_list

    def find_kernel_by_name(self, name):
        matches = [k for k in self.kernel_list if k.name == name]
        assert len(matches) < 2, "find_kernel_by_name(): More than one match for kernel name!"
        return matches[0] if len(matches) == 1 else None

    def ndims(self):
        return self.dims

    def add_property(self, prop_name, prop_type, value=0.0, vol=False):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        return self.properties.add(prop_name, prop_type, value, vol)

    def add_position(self, prop_name, value=[0.0, 0.0, 0.0], vol=False, layout=Layouts.AoS):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        self.position_prop = self.properties.add(prop_name, Types.Vector, value, vol, layout)
        return self.position_prop

    def add_feature(self, feature_name, nkinds):
        assert self.feature(feature_name) is None, f"Feature already defined: {feature_name}"
        return self.features.add(feature_name, nkinds)

    def add_feature_property(self, feature_name, prop_name, prop_type, prop_data):
        feature = self.feature(feature_name)
        assert feature is not None, f"Feature not found: {feature_name}"
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        return self.feature_properties.add(feature, prop_name, prop_type, prop_data)

    def add_contact_property(self, prop_name, prop_type, prop_default, layout=Layouts.AoS):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        return self.contact_properties.add(prop_name, prop_type, layout, prop_default)

    def property(self, prop_name):
        return self.properties.find(prop_name)

    def position(self):
        return self.position_prop

    def feature(self, feature_name):
        return self.features.find(feature_name)

    def feature_property(self, feature_prop_name):
        return self.feature_properties.find(feature_prop_name)

    def contact_property(self, contact_prop_name):
        return self.contact_properties.find(contact_prop_name)

    def add_array(self, arr_name, arr_sizes, arr_type, arr_layout=Layouts.AoS, arr_sync=True):
        assert self.array(arr_name) is None, f"Array already defined: {arr_name}"
        return self.arrays.add(arr_name, arr_sizes, arr_type, arr_layout, arr_sync)

    def add_static_array(self, arr_name, arr_sizes, arr_type, arr_layout=Layouts.AoS, init_value=None):
        assert self.array(arr_name) is None, f"Array already defined: {arr_name}"
        return self.arrays.add_static(arr_name, arr_sizes, arr_type, arr_layout, init_value=init_value)

    def array(self, arr_name):
        return self.arrays.find(arr_name)

    def add_var(self, var_name, var_type, init_value=0):
        assert self.var(var_name) is None, f"Variable already defined: {var_name}"
        return self.vars.add(var_name, var_type, init_value)

    def add_temp_var(self, init_value):
        return self.vars.add_temp(init_value)

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

    def read_particle_data(self, filename, prop_names):
        props = [self.property(prop_name) for prop_name in prop_names]
        read_object = ReadParticleData(self, filename, props)
        self.setups.add_statement(read_object)
        self.grid = read_object.grid

    def build_cell_lists(self, spacing):
        self.cell_lists = CellLists(self, self.grid, spacing, spacing)
        return self.cell_lists

    def build_neighbor_lists(self, spacing):
        self.cell_lists = CellLists(self, self.grid, spacing, spacing)
        self.neighbor_lists = NeighborLists(self.cell_lists)
        return self.neighbor_lists

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

    def build_module_with_statements(self, run_on_device=True):
        self.functions.add_statement(
            Module(self,
                name=self._module_name,
                block=Block(self, self._block),
                resizes_to_check=self._resizes_to_check,
                check_properties_resize=self._check_properties_resize,
                run_on_device=run_on_device))

    def capture_statements(self, capture=True):
        self._capture_statements = capture

    def add_statement(self, stmt):
        if self._capture_statements:
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
        self.code_gen.assign_target(target)

    def cell_spacing(self):
        return self.cell_lists.cutoff_radius

    def domain_partitioning(self):
        return self._dom_part

    def generate(self):
        assert self._target is not None, "Target not specified!"
        comm = Comm(self, self._dom_part)

        timestep_procedures = [
            (comm.exchange(), 20),
            (comm.borders(), comm.synchronize(), 20),
            (CellListsBuild(self, self.cell_lists), 20),
            (NeighborListsBuild(self, self.neighbor_lists), 20),
        ]

        if not self.contact_properties.empty():
            contact_history = ContactHistory(self.cell_lists)
            timestep_procedures.append((BuildContactHistory(self, contact_history), 20))

        timestep_procedures += [
            PropertiesResetVolatile(self),
            self.functions
        ]

        timestep = Timestep(self, self.ntimesteps, timestep_procedures)
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
            ContactPropertiesAlloc(self),
            FeaturePropertiesAlloc(self)
        ])

        program = Module(self, name='main', block=Block.merge_blocks(decls, body))

        # Apply transformations
        transformations = Transformations(program, self._target)
        transformations.apply_all()

        # Generate program
        #ASTGraph(self.functions, "functions.dot").render()
        self.code_gen.generate_program(program)
