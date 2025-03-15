from pairs.ir.arrays import Arrays
from pairs.ir.block import Block
from pairs.ir.features import Features, FeatureProperties
from pairs.ir.kernel import Kernel
from pairs.ir.layouts import Layouts
from pairs.ir.module import Module
from pairs.ir.properties import Properties, ContactProperties
from pairs.ir.symbols import Symbol
from pairs.ir.types import Types
from pairs.ir.variables import Variables
#from pairs.graph.graphviz import ASTGraph
from pairs.mapping.funcs import compute
from pairs.sim.cell_lists import CellLists, BuildCellLists, PartitionCellLists, BuildCellNeighborLists
from pairs.sim.comm import Comm
from pairs.sim.contact_history import ContactHistory
from pairs.sim.domain_partitioners import DomainPartitioners
from pairs.sim.domain_partitioning import BlockForest, DimensionRanges
from pairs.sim.grid import Grid3D
from pairs.sim.neighbor_lists import NeighborLists, BuildNeighborLists
from pairs.transformations import Transformations
from pairs.code_gen.interface import InterfaceModules


class Simulation:
    """P4IRS Simulation class, this class is the center of kernel simulations which contains all
       fundamental data structures to generate a P4IRS simulation code"""
    def __init__(
        self,
        code_gen,
        shapes,
        dims=3,
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
        self.particle_uid = self.add_property('uid', Types.UInt64, 0)
        self.particle_shape = self.add_property('shape', Types.Int32, 0)
        self.particle_flags = self.add_property('flags', Types.Int32, 0)

        # Grid for the simulation
        self.grid = None

        # Acceleration structures
        self._use_halo_cells = False
        self.cell_lists = None
        self._store_neighbors_per_cell = False
        self.neighbor_lists = None
        self.update_cells_procedures = Block(self, [])

        # Context information used to partially build the program AST
        self.scope = []
        self.nested_count = 0
        self.nest = False
        self._capture_statements = True
        self._block = Block(self, [])

        # Interface modules
        self.interface_module_list = []
        
        # Internal modules and kernels
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
        self._comm = None

        # Contact history
        self._use_contact_history = use_contact_history
        self._contact_history = ContactHistory(self) if use_contact_history else None


        self._module_name = None                # Current module name
        self._double_prec = double_prec         # Use double-precision FP arithmetic
        self.dims = dims                        # Number of dimensions
        self.reneighbor_frequency = 1           # Re-neighbor frequency
        self.rebalance_frequency = 0            # Re-balance frequency for dynamic load balancing
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

    def add_interface_module(self, module):
        assert isinstance(module, Module), "add_interface_module(): Given parameter is not of type Module!"
        assert module.interface
        if module.name not in [m.name for m in self.interface_module_list]:
            self.interface_module_list.append(module)

    def add_module(self, module):
        assert isinstance(module, Module), "add_module(): Given parameter is not of type Module!"
        assert not module.interface
        if module.name not in [m.name for m in self.module_list]:
            self.module_list.append(module)

    def interface_modules(self):
        """List interface modules"""
        return self.interface_module_list
        
    def modules(self):
        """List internal modules"""
        return self.module_list

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

    def add_property(self, prop_name, prop_type, value=0.0, volatile=False, reduce=False):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        return self.properties.add(prop_name, prop_type, value, volatile, p_reduce=reduce)

    def add_position(self, prop_name, value=[0.0, 0.0, 0.0], volatile=False, layout=Layouts.AoS):
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"
        self.position_prop = self.properties.add(prop_name, Types.Vector, value, volatile, layout)
        return self.position_prop

    def add_feature(self, feature_name, nkinds):
        assert self.feature(feature_name) is None, f"Feature already defined: {feature_name}"
        return self.features.add(feature_name, nkinds)

    def add_feature_property(self, feature_name, prop_name, prop_type, prop_data=None):
        feature = self.feature(feature_name)
        assert feature is not None, f"Feature not found: {feature_name}"
        assert self.property(prop_name) is None, f"Property already defined: {prop_name}"

        array_size = feature.nkinds()**2 * Types.number_of_elements(self, prop_type)

        if not prop_data:
            prop_data = [0 for i in range(array_size)]
        else:
            assert len(prop_data) == array_size, f"Incorrect array size for {prop_name}: Expected array size = {array_size}"

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
        return self.vars.add(var_name, var_type, init_value, runtime)

    def add_temp_var(self, init_value, type=None):
        return self.vars.add_temp(init_value, type)

    def add_symbol(self, sym_type, name=None):
        return Symbol(self, sym_type, name)

    def var(self, var_name):
        return self.vars.find(var_name)

    def set_domain(self, grid):
        """Set domain bounds if they are known at P4IRS compile time"""
        self.grid = Grid3D(self, grid[0], grid[1], grid[2], grid[3], grid[4], grid[5])

    def reneighbor_every(self, frequency):
        self.reneighbor_frequency = frequency

    def build_cell_lists(self, spacing=None, store_neighbors_per_cell=False, use_halo_cells=False):
        """Add routines to build the linked-cells acceleration structure.
        Leave spacing as None so it can be set at runtime."""
        self._store_neighbors_per_cell = store_neighbors_per_cell
        self._use_halo_cells = use_halo_cells
        self.cell_lists = CellLists(self, self._dom_part, spacing, spacing)
        return self.cell_lists

    def build_neighbor_lists(self, spacing=None):
        """Add routines to build the Verlet Lists acceleration structure.
        Leave spacing as None so it can be set at runtime."""

        assert self._store_neighbors_per_cell is False, \
            "Using neighbor-lists with store_neighbors_per_cell option is invalid."

        self.cell_lists = CellLists(self, self._dom_part, spacing, spacing)
        self.neighbor_lists = NeighborLists(self, self.cell_lists)
        return self.neighbor_lists

    def compute(self, func, cutoff_radius=None, symbols={}, parameters={}, compute_globals=False, run_on_device=True, profile=False):
        return compute(self, func, cutoff_radius, symbols, parameters, compute_globals, run_on_device, profile)

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

    def build_interface_module_with_statements(self):
        """Build a user-defined Module that will be callable seperately as part of the interface"""
        Module(self, name=self._module_name,
                block=Block(self, self._block),
                resizes_to_check=self._resizes_to_check,
                check_properties_resize=self._check_properties_resize,
                interface=True)
        
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

    def create_update_cells_block(self):
        subroutines = [
            BuildCellLists(self, self.cell_lists),
            PartitionCellLists(self, self.cell_lists)
        ]

        # Add routine to build neighbor-lists per cell
        if self._store_neighbors_per_cell:
            subroutines.append(BuildCellNeighborLists(self, self.cell_lists))

        # Add routine to build neighbor-lists per particle (standard Verlet Lists)
        if self.neighbor_lists is not None:
            subroutines.append(BuildNeighborLists(self, self.neighbor_lists))

        self.update_cells_procedures.add_statement(subroutines)

    def generate(self):
        """Generate the code for the simulation"""
        assert self._target is not None, "Target not specified!"

        # Initialize communication instance with the specified domain-partitioner
        self._comm = Comm(self, self._dom_part)
        self.create_update_cells_block()

        InterfaceModules(self).create_all()
        Transformations(self.interface_modules(), self._target).apply_all()

        # Generate library
        self.code_gen.generate_library()

        # Generate getters for the runtime functions
        self.code_gen.generate_interfaces()