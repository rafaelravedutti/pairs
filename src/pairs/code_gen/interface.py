from pairs.ir.block import Block, pairs_interface_block
from pairs.ir.functions import Call_Void, Call, Call_Int
from pairs.ir.parameters import Parameter
from pairs.ir.ret import Return
from pairs.ir.scalars import ScalarOp
from pairs.ir.types import Types
from pairs.ir.branches import Filter, Branch
from pairs.ir.select import Select
from pairs.ir.atomic import AtomicInc
from pairs.ir.properties import ReallocProperty
from pairs.ir.print import PrintCode
from pairs.ir.assign import Assign
from pairs.sim.flags import Flags
from pairs.sim.domain import DomainRebalance, DomainUpdateLocal, DomainUpdateNeighborhood
from pairs.sim.cell_lists import BuildCellListsStencil
from pairs.sim.comm import Synchronize, Borders, Exchange, ReverseComm
from pairs.sim.cell_lists import BuildCellLists, BuildCellListsStencil, PartitionCellLists, BuildCellNeighborLists
from pairs.sim.neighbor_lists import BuildNeighborLists
from pairs.sim.variables import DeclareVariables 
from pairs.sim.arrays import DeclareArrays
from pairs.sim.properties import AllocateProperties, AllocateContactProperties, ResetVolatileProperties
from pairs.sim.features import AllocateFeatureProperties
from pairs.sim.instrumentation import RegisterMarkers, RegisterTimers
from pairs.sim.grid import MutableGrid
from pairs.sim.domain_partitioners import DomainPartitioners
from pairs.sim.contact_history import BuildContactHistory, ClearUnusedContactHistory, ResetContactHistoryUsageStatus

class InterfaceModules:
    def __init__(self, sim):
        self.sim = sim

    def create_all(self):
        self.initialize()
        self.setCellWidth()
        self.setInteractionRadius()
        self.updateDomain()
        self.reneighbor()
        self.refreshGhosts() 
        self.reverseCommunicate() 
        self.resetVolatiles()
        self.rank()
        self.nlocal()
        self.nghost()
        self.size()
        self.end()      
        self.create_object()

    @pairs_interface_block
    def initialize(self):
        self.sim.module_name('initialize')
        nprops = self.sim.properties.nprops()
        ncontactprops = self.sim.contact_properties.nprops()
        narrays = self.sim.arrays.narrays()
        part = DomainPartitioners.c_keyword(self.sim.partitioner())

        PrintCode(self.sim, f"pairs_runtime = new PairsRuntime({nprops}, {ncontactprops}, {narrays}, {part});")
        PrintCode(self.sim, f"pobj = new PairsObjects();")

        if self.sim.grid is None:
            self.sim.grid = MutableGrid(self.sim, self.sim.dims)

        inits = Block.from_list(self.sim, [
            RegisterTimers(self.sim),
            RegisterMarkers(self.sim),
            DeclareVariables(self.sim),
            DeclareArrays(self.sim),
            AllocateProperties(self.sim),
            AllocateContactProperties(self.sim),
            AllocateFeatureProperties(self.sim),
        ])

        if self.sim._enable_profiler:
            PrintCode(self.sim, "LIKWID_MARKER_INIT;")

        self.sim.add_statement(inits)

    @pairs_interface_block
    def setCellWidth(self):
        self.sim.module_name('setCellWidth')
        if self.sim.cell_lists.runtime_spacing:
            for d in range(self.sim.dims):
                Assign(self.sim, self.sim.cell_lists.spacing[d], Parameter(self.sim, f'cell_width_dim_{d}', Types.Real))

    @pairs_interface_block
    def setInteractionRadius(self):
        self.sim.module_name('setInteractionRadius')
        if self.sim.cell_lists.runtime_cutoff_radius:
            Assign(self.sim, self.sim.cell_lists.cutoff_radius, Parameter(self.sim, 'cutoff_radius', Types.Real))
        
    @pairs_interface_block
    def updateDomain(self):
        ''' This function is required to be called only once after all particles have been created.
        If rebalancing is enabled, the domain is rebalanced everytime this function is called.
        If rebalancing is disabled, calling this function has the same effect as calling 'reneighbor'. 
        '''
        self.sim.module_name('updateDomain')

        self.sim.add_statement(DomainUpdateNeighborhood(self.sim)) 

        # Local particles must be contained in their owners before rebalancing, otherwise they may get lost
        self.sim.add_statement(Exchange(self.sim))

        # Here AABBs assigned to each rank may change if rebalancing is enabled
        self.sim.add_statement(DomainRebalance(self.sim)) 

        # This is a cheap update to crop the subdom and find local non-empty AABBs
        # Note: All local particles are strictly contained within AABBs. Therefore, 
        # no padding is needed to find non-empty AABBs
        self.sim.add_statement(DomainUpdateLocal(self.sim))      

        # Rebuild stencil since subdom sizes have changed. Also may use non-empty AABBs to create halo cells
        self.sim.add_statement(BuildCellListsStencil(self.sim)) 

        if self.sim._use_halo_cells:
            self.sim.add_statement(BuildCellLists(self, self.cell_lists))   
            self.sim.add_statement(Borders(self.sim))

        else: 
            # Exchange is not needed since all locals are contained in thier owners after deserialization
            # But ghosts must be recreated after rebalancing (optionally uses the halo cells)
            self.sim.add_statement(Borders(self.sim))

        # Populate cells with local and ghost particles
        self.sim.add_statement(self.sim.update_cells_procedures)   


        # Reset volatile includes the new locals
        self.sim.add_statement(ResetVolatileProperties(self.sim))  

    @pairs_interface_block
    def reneighbor(self):
        self.sim.module_name('reneighbor')

        reneighboring_procedures = [
            Exchange(self.sim),
            # Note: DomainUpdateLocal must happen after exchange since local particles must be contained in AABBs.
            #       And it must happen before Borders since newly received particles need to be included, so they become ghosts
            #       for their previous neighbor
            DomainUpdateLocal(self.sim),    
            Borders(self.sim),
            BuildCellListsStencil(self.sim),
            self.sim.update_cells_procedures,
            ResetVolatileProperties(self.sim)
        ]

        if self.sim._use_contact_history:
            reneighboring_procedures += [
                BuildContactHistory(self.sim, self.sim._contact_history, self.sim.cell_lists),
                ResetContactHistoryUsageStatus(self.sim, self.sim._contact_history),
                ClearUnusedContactHistory(self.sim, self.sim._contact_history)
            ]
        
        self.sim.add_statement(reneighboring_procedures)

    @pairs_interface_block
    def refreshGhosts(self):
        self.sim.module_name('refreshGhosts')
        self.sim.add_statement(Synchronize(self.sim))

    @pairs_interface_block
    def reverseCommunicate(self):
        self.sim.module_name('reverseCommunicate')
        self.sim.add_statement(ReverseComm(self.sim, reduce=True))
    
    @pairs_interface_block
    def resetVolatiles(self):
        self.sim.module_name('resetVolatileProperties')
        self.sim.add_statement(ResetVolatileProperties(self.sim))
    
    @pairs_interface_block
    def rank(self):
        self.sim.module_name('rank')
        Return(self.sim, self.sim.domain_partitioning().rank)

    @pairs_interface_block
    def nlocal(self):
        self.sim.module_name('nlocal')
        Return(self.sim, self.sim.nlocal)

    @pairs_interface_block
    def nghost(self):
        self.sim.module_name('nghost')
        Return(self.sim, self.sim.nghost)

    @pairs_interface_block
    def size(self):
        self.sim.module_name('size')
        Return(self.sim, ScalarOp.inline(self.sim.nlocal + self.sim.nghost))

    @pairs_interface_block
    def end(self):
        self.sim.module_name('end')

        if self.sim._enable_profiler:
            PrintCode(self.sim, "LIKWID_MARKER_CLOSE;")
            
        Call_Void(self.sim, "pairs::print_timers", [])
        # Call_Void(self.sim, "pairs::log_timers", [])
        Call_Void(self.sim, "pairs::print_stats", [self.sim.nlocal, self.sim.nghost])
        PrintCode(self.sim, "delete pobj;")
        PrintCode(self.sim, "delete pairs_runtime;")

    @pairs_interface_block
    def create_object(self):
        self.sim.module_name('createObject')

        # Initialize idx with an invalid value
        idx = self.sim.add_temp_var(-1)

        # Define function parameters (position coordinates, shape and flags)
        pos = []
        ndims = self.sim.ndims()
        for d in range(ndims):
            pos.append(Parameter(self.sim, f'pos_dim_{d}', Types.Real, d))

        shape = Parameter(self.sim, 'shape', Types.Int32, ndims)
        flags = Parameter(self.sim, 'flags', Types.Int32, ndims + 1)

        # Check if the provided position is within this subdomain (i.e. particle is local)
        is_local = Call(self.sim, "pairs_runtime->getDomainPartitioner()->isWithinSubdomain", pos, Types.Boolean)
        
        # Check if global or infinite flag is set for the provided flag
        is_global = flags & (Flags.Global | Flags.Infinite)
        
        # If the particle is local or global
        for _ in Filter(self.sim, ScalarOp.or_op(is_local, is_global)):
            # Set idx to the current nlocal
            Assign(self.sim, idx, self.sim.nlocal)

            # If idx exceeds particle capacity
            for _ in Filter(self.sim, self.sim.particle_capacity <= idx):
                # Resize capacity 
                Assign(self.sim, self.sim.particle_capacity, idx * 2)

                # Reallocate properties
                for p in self.sim.properties.all():
                    sizes = [self.sim.particle_capacity] if Types.is_scalar(p.type()) else \
                            [self.sim.particle_capacity, Types.number_of_elements(self.sim, p.type())]
                    ReallocProperty(self.sim, p, sizes)

                # Reallocate arrays bonded to this capacity
                for arr in self.sim.particle_capacity.bonded_arrays():
                    arr.realloc()

            # Set particle position
            for d in range(ndims):
                Assign(self.sim, self.sim.particle_position[idx][d], pos[d])
            
            # Set particle shape
            Assign(self.sim, self.sim.particle_shape[idx], shape)

            # Set particle flags
            Assign(self.sim, self.sim.particle_flags[idx], flags)
            
            # Set particle UID
            # (creates a global UID or a local UID of type uint64_t according to the flags provided)
            Assign(self.sim, self.sim.particle_uid[idx], 
                        Select(self.sim, is_global, 
                                Call(self.sim, "UniqueID::createGlobal", [], Types.UInt64),  
                                Call(self.sim, "UniqueID::create", [], Types.UInt64)))
            
            # Increment nlocal
            Assign(self.sim, self.sim.nlocal, self.sim.nlocal + 1)
        
        # Return the idx
        Return(self.sim, idx)
