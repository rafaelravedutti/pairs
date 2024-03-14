from pairs.ir.arrays import Arrays
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.features import Features, FeatureProperties
from pairs.ir.kernel import Kernel
from pairs.ir.layouts import Layouts
from pairs.ir.module import Module
from pairs.ir.properties import Properties, ContactProperties
from pairs.ir.symbols import Symbol
from pairs.ir.types import Types
from pairs.ir.variables import Variables
#from pairs.graph.graphviz import ASTGraph
from pairs.mapping.funcs import compute, setup
from pairs.sim.arrays import DeclareArrays
from pairs.sim.cell_lists import CellLists, BuildCellLists, BuildCellListsStencil, PartitionCellLists, BuildCellNeighborLists
from pairs.sim.comm import Comm
from pairs.sim.contact_history import ContactHistory, BuildContactHistory, ClearUnusedContactHistory, ResetContactHistoryUsageStatus
from pairs.sim.copper_fcc_lattice import CopperFCCLattice
from pairs.sim.dem_sc_grid import DEMSCGrid
from pairs.sim.domain import InitializeDomain
from pairs.sim.domain_partitioners import DomainPartitioners
from pairs.sim.domain_partitioning import BlockForest, DimensionRanges
from pairs.sim.features import AllocateFeatureProperties
from pairs.sim.grid import Grid2D, Grid3D
from pairs.sim.instrumentation import RegisterMarkers, RegisterTimers
from pairs.sim.lattice import ParticleLattice
from pairs.sim.neighbor_lists import NeighborLists, BuildNeighborLists
from pairs.sim.properties import AllocateProperties, AllocateContactProperties, ResetVolatileProperties
from pairs.sim.read_from_file import ReadParticleData
from pairs.sim.thermo import ComputeThermo
from pairs.sim.timestep import Timestep
from pairs.sim.variables import DeclareVariables 
from pairs.sim.vtk import VTKWrite
from pairs.transformations import Transformations


class Simulation:
    """P4IRS Simulation class, this class is the center of kernel simulations which contains all
       fundamental data structures to generate a P4IRS simulation code"""
    def __init__(
        self,
        code_gen,
        shapes,
        dims=3,
        timesteps=100,
        double_prec=False,
        use_contact_history=False,
        particle_capacity=800000,
        neighbor_capacity=100):

        # Code generator for the simulation
        self.code_gen = code_gen
        self.code_gen.assign_simulation(self)
        # Data structures to be generated
        self.position_prop = None
        self.properties = Properties(self)
        self.vars = Variables(self)
        self.arrays = Arrays(self)
        self.features = Features(self)
        self.feature_properties = FeatureProperties(self)
        self.contact_properties = ContactProperties(self)

        # General capacities, sizes and particle properties
        self.particle_capacity = \
            self.add_var('particle_capacity', Types.Int32, particle_capacity, runtime=True)
        self.neighbor_capacity = self.add_var('neighbor_capacity', Types.Int32, neighbor_capacity)
        self.nlocal = self.add_var('nlocal', Types.Int32, runtime=True)
        self.nghost = self.add_var('nghost', Types.Int32, runtime=True)
        self.resizes = self.add_array('resizes', 3, Types.Int32, arr_sync=False)
        self.particle_uid = self.add_property('uid', Types.Int32, 0)
        self.particle_shape = self.add_property('shape', Types.Int32, 0)
        self.particle_flags = self.add_property('flags', Types.Int32, 0)

        # Grid for the simulation
        self.grid = None

        # Acceleration structures
        self.cell_lists = None
        self._store_neighbors_per_cell = False
        self.neighbor_lists = None

        # Context information used to partially build the program AST
        self.scope = []
        self.nested_count = 0
        self.nest = False
        self._capture_statements = True
        self._block = Block(self, [])

        # Different segments of particle code/functions
        self.setups = Block(self, [])
        self.setup_functions = []
        self.pre_step_functions = []
        self.functions = []
        self.module_list = []
        self.kernel_list = []

        # Structures to generated resize code for capacities
        self._check_properties_resize = False
        self._resizes_to_check = {}

        # VTK data
        self.vtk_file = None
        self.vtk_frequency = 0

        # Domain partitioning
        self._dom_part = None
        self._partitioner = None

        # Contact history
        self._use_contact_history = use_contact_history
        self._contact_history = ContactHistory(self) if use_contact_history else None


        self._module_name = None                # Current module name
        self._double_prec = double_prec         # Use double-precision FP arithmetic
        self.dims = dims                        # Number of dimensions
        self.ntimesteps = timesteps             # Number of time-steps
        self.reneighbor_frequency = 1           # Re-neighbor frequency
        self._target = None                     # Hardware target info
        self._pbc = [True for _ in range(dims)] # PBC flags for each dimension
        self._shapes = shapes                   # List of shapes used in the simulation
        self._compute_half = False              # Compute half of interactions (Newton 3D Law)
        self._apply_list = None                 # Context elements when using apply() directive
        self._enable_profiler = False           # Enable/disable profiler
        self._compute_thermo = 0                # Compute thermo information

    def set_domain_partitioner(self, partitioner):
        """Selects domain-partitioner used and create its object for this simulation instance"""
        self._partitioner = partitioner

        if partitioner in (DomainPartitioners.Regular, DomainPartitioners.RegularXY):
            self._dom_part = DimensionRanges(self)

        elif partitioner == DomainPartitioners.BlockForest:
            self._dom_part = BlockForest(self)

        else:
            raise Exception("Invalid domain partitioner.")

    def partitioner(self):
        return self._partitioner

    def enable_profiler(self):
        self._enable_profiler = True

    def compute_half(self):
        self._compute_half = True

    def use_double_precision(self):
        return self._double_prec

    def get_shape_id(self, shape):
        return self._shapes[shape]

    def max_shapes(self):
        return len(self._shapes)

    def add_module(self, module):
        assert isinstance(module, Module), "add_module(): Given parameter is not of type Module!"
        if module.name not in [m.name for m in self.module_list]:
            self.module_list.append(module)

    def modules(self):
        """List simulation modudles, with main always in the last position"""

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

    def pbc(self, pbc_config):
        assert len(pbc_config) == self.dims, "PBC must be specified for each dimension."
        self._pbc = pbc_config

    def add_property(self, prop_name, prop_type, value=0.0, volatile=False):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        return self.properties.add(prop_name, prop_type, value, volatile)

    def add_position(self, prop_name, value=[0.0, 0.0, 0.0], volatile=False, layout=Layouts.AoS):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        self.position_prop = self.properties.add(prop_name, Types.Vector, value, volatile, layout)
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

    def add_var(self, var_name, var_type, init_value=0, runtime=False):
        assert self.var(var_name) is None, f"Variable already defined: {var_name}"
        return self.vars.add(var_name, var_type, init_value)

    def add_temp_var(self, init_value):
        return self.vars.add_temp(init_value)

    def add_symbol(self, sym_type):
        return Symbol(self, sym_type)

    def var(self, var_name):
        return self.vars.find(var_name)

    def set_domain(self, grid):
        self.grid = Grid3D(self, grid[0], grid[1], grid[2], grid[3], grid[4], grid[5])
        self.setups.add_statement(InitializeDomain(self))

    def reneighbor_every(self, frequency):
        self.reneighbor_frequency = frequency

    def create_particle_lattice(self, grid, spacing, props={}):
        self.setups.add_statement(ParticleLattice(self, grid, spacing, props, self.position()))

    def read_particle_data(self, filename, prop_names, shape_id):
        """Generate statement to read particle data from file"""
        props = [self.property(prop_name) for prop_name in prop_names]
        self.setups.add_statement(ReadParticleData(self, filename, props, shape_id))

    def copper_fcc_lattice(self, nx, ny, nz, rho, temperature, ntypes):
        """Specific initialization for MD Copper FCC lattice case"""
        self.setups.add_statement(CopperFCCLattice(self, nx, ny, nz, rho, temperature, ntypes))

    def dem_sc_grid(self, xmax, ymax, zmax, spacing, diameter, min_diameter, max_diameter, initial_velocity, particle_density, ntypes):
        """Specific initialization for DEM grid"""
        self.setups.add_statement(
            DEMSCGrid(self, xmax, ymax, zmax, spacing, diameter, min_diameter, max_diameter,
                      initial_velocity, particle_density, ntypes))

    def build_cell_lists(self, spacing, store_neighbors_per_cell=False):
        """Add routines to build the linked-cells acceleration structure"""
        self._store_neighbors_per_cell = store_neighbors_per_cell
        self.cell_lists = CellLists(self, self._dom_part, spacing, spacing)
        return self.cell_lists

    def build_neighbor_lists(self, spacing):
        """Add routines to build the Verlet Lists acceleration structure"""

        assert self._store_neighbors_per_cell is False, \
            "Using neighbor-lists with store_neighbors_per_cell option is invalid."

        self.cell_lists = CellLists(self, self._dom_part, spacing, spacing)
        self.neighbor_lists = NeighborLists(self, self.cell_lists)
        return self.neighbor_lists

    def compute(self, func, cutoff_radius=None, symbols={}, pre_step=False, skip_first=False):
        return compute(self, func, cutoff_radius, symbols, pre_step, skip_first)

    def setup(self, func, symbols={}):
        return setup(self, func, symbols)

    def init_block(self):
        """Initialize new block in this simulation instance"""
        self._block = Block(self, [])
        self._check_properties_resize = False
        self._resizes_to_check = {}
        self._module_name = None

    def module_name(self, name):
        self._module_name = name

    def check_properties_resize(self):
        """Enable checking properties for resizing"""
        self._check_properties_resize = True

    def check_resize(self, capacity, size):
        """Determine that capacity must always be checked with respect to size in a block/module"""

        if capacity not in self._resizes_to_check:
            self._resizes_to_check[capacity] = size
        else:
            raise Exception("Two sizes assigned to same capacity!")

    def build_setup_module_with_statements(self):
        """Build a Module in the setup part of the program using the last initialized block"""

        self.setup_functions.append(
            Module(self,
                name=self._module_name,
                block=Block(self, self._block),
                resizes_to_check=self._resizes_to_check,
                check_properties_resize=self._check_properties_resize,
                run_on_device=False))

    def build_pre_step_module_with_statements(self, run_on_device=True, skip_first=False, profile=False):
        """Build a Module in the pre-step part of the program using the last initialized block"""
        module = Module(self, name=self._module_name,
                              block=Block(self, self._block),
                              resizes_to_check=self._resizes_to_check,
                              check_properties_resize=self._check_properties_resize,
                              run_on_device=run_on_device)

        if profile:
            module.profile()

        if skip_first:
            self.pre_step_functions.append((module, {'skip_first': True}))

        else:
            self.pre_step_functions.append(module)

    def build_module_with_statements(self, run_on_device=True, skip_first=False, profile=False):
        """Build a Module in the compute part of the program using the last initialized block"""
        module = Module(self, name=self._module_name,
                              block=Block(self, self._block),
                              resizes_to_check=self._resizes_to_check,
                              check_properties_resize=self._check_properties_resize,
                              run_on_device=run_on_device)
        if profile:
            module.profile()

        if skip_first:
            self.functions.append((module, {'skip_first': True}))

        else:
            self.functions.append(module)

    def capture_statements(self, capture=True):
        """When toggled, all constructed statements are captured and automatically added to the last initialized block"""
        self._capture_statements = capture

    def add_statement(self, stmt):
        """Add captured statements to the last block when _capture_statements is toggled"""
        if self._capture_statements:
            if not self.scope:
                self._block.add_statement(stmt)
            else:
                self.scope[-1].add_statement(stmt)

        return stmt

    def nest_mode(self):
        """When explicitly constructing loops in P4IRS, make them nested"""
        self.nested_count = 0
        self.nest = True
        yield
        self.nest = False
        for _ in range(0, self.nested_count):
            self.scope.pop()

    def enter(self, scope):
        """Enter a new scope, used for tracking scopes when building P4IRS AST elements"""
        self.scope.append(scope)

    def leave(self):
        """Leave last scope, used for tracking scopes when building P4IRS AST elements"""
        if not self.nest:
            self.scope.pop()
        else:
            self.nested_count += 1

    def use_apply_list(self, apply_list):
        self._apply_list = apply_list

    def release_apply_list(self):
        self._apply_list = None

    def current_apply_list(self):
        return self._apply_list

    def vtk_output(self, filename, frequency=0):
        self.vtk_file = filename
        self.vtk_frequency = frequency

    def target(self, target):
        self._target = target
        self.code_gen.assign_target(target)

    def cell_spacing(self):
        return self.cell_lists.cutoff_radius

    def domain_partitioning(self):
        return self._dom_part

    def compute_thermo(self, every=0):
        self._compute_thermo = every

    def generate(self):
        """Generate the code for the simulation"""

        assert self._target is not None, "Target not specified!"

        # Initialize communication instance with specified domain-partitioner
        comm = Comm(self, self._dom_part)
        # Params that determine when a method must be called only when reneighboring
        every_reneighbor_params = {'every': self.reneighbor_frequency}

        # First steps executed during each time-step in the simulation
        timestep_procedures = self.pre_step_functions + [
            (comm.exchange(), every_reneighbor_params),
            (comm.borders(), comm.synchronize(), every_reneighbor_params),
            (BuildCellLists(self, self.cell_lists), every_reneighbor_params),
            (PartitionCellLists(self, self.cell_lists), every_reneighbor_params)
        ]

        # Add routine to build neighbor-lists per cell
        if self._store_neighbors_per_cell:
            timestep_procedures.append(
                (BuildCellNeighborLists(self, self.cell_lists), every_reneighbor_params))

        # Add routine to build neighbor-lists per particle (standard Verlet Lists)
        if self.neighbor_lists is not None:
            timestep_procedures.append(
                (BuildNeighborLists(self, self.neighbor_lists), every_reneighbor_params))

        # Add routines for contact history management
        if self._use_contact_history:
            if self.neighbor_lists is not None:
                timestep_procedures.append(
                    (BuildContactHistory(self, self._contact_history, self.cell_lists),
                    every_reneighbor_params))

            timestep_procedures.append(ResetContactHistoryUsageStatus(self, self._contact_history))

        # Reset volatile properties and add computational kernels
        timestep_procedures += [ResetVolatileProperties(self)] + self.functions

        # Clear unused contact history
        if self._use_contact_history:
            timestep_procedures.append(ClearUnusedContactHistory(self, self._contact_history))

        # Add routine to calculate thermal data
        if self._compute_thermo != 0:
            timestep_procedures.append(
                (ComputeThermo(self), {'every': self._compute_thermo}))

        # Construct the time-step loop
        timestep = Timestep(self, self.ntimesteps, timestep_procedures)
        self.enter(timestep.block)

        # Add routine to write VTK data when set
        if self.vtk_file is not None:
            timestep.add(VTKWrite(self, self.vtk_file, timestep.timestep(), self.vtk_frequency))

        self.leave()

        # Initialization and setup functions, together with time-step loop
        body = Block.from_list(self, [
            self.setups,
            self.setup_functions,
            BuildCellListsStencil(self, self.cell_lists),
            timestep.as_block()
        ])

        # Data structures and timer/markers initialization
        inits = Block.from_list(self, [
            DeclareVariables(self),
            DeclareArrays(self),
            AllocateProperties(self),
            AllocateContactProperties(self),
            AllocateFeatureProperties(self),
            RegisterTimers(self),
            RegisterMarkers(self)
        ])

        # Combine everything into a whole program
        program = Module(self, name='main', block=Block.merge_blocks(inits, body))

        # Apply transformations
        transformations = Transformations(program, self._target)
        transformations.apply_all()

        # Generate program
        #ASTGraph(self.functions, "functions.dot").render()
        self.code_gen.generate_program(program)
        self.code_gen.generate_interfaces()
