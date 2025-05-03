from pairs.ir.block import Block, pairs_interface_block
from pairs.ir.functions import Call_Void, Call, Call_Int
from pairs.ir.parameters import Parameter
from pairs.ir.ret import Return
from pairs.ir.scalars import ScalarOp
from pairs.sim.domain import DomainRebalance, DomainUpdateLocal, DomainUpdateNeighborhood
from pairs.sim.cell_lists import BuildCellListsStencil
from pairs.sim.comm import Synchronize, Borders, Exchange, ReverseComm
from pairs.ir.types import Types
from pairs.ir.branches import Filter, Branch
from pairs.sim.cell_lists import BuildCellLists, BuildCellListsStencil, PartitionCellLists, BuildCellNeighborLists
from pairs.sim.neighbor_lists import BuildNeighborLists
from pairs.sim.variables import DeclareVariables 
from pairs.sim.arrays import DeclareArrays
from pairs.sim.properties import AllocateProperties, AllocateContactProperties, ResetVolatileProperties
from pairs.sim.features import AllocateFeatureProperties
from pairs.sim.instrumentation import RegisterMarkers, RegisterTimers
from pairs.sim.grid import MutableGrid
from pairs.sim.domain_partitioners import DomainPartitioners
from pairs.ir.print import PrintCode
from pairs.ir.assign import Assign
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
        self.sim.add_statement(Exchange(self.sim._comm))

        # Here AABBs assigned to each rank may change if rebalancing is enabled
        self.sim.add_statement(DomainRebalance(self.sim)) 

        # This is a cheap update to crop the subdom and find local non-empty AABBs
        # Note: All local particles are strictly contained within AABBs. Therefore, 
        # no padding is needed to find non-empty AABBs
        self.sim.add_statement(DomainUpdateLocal(self.sim))      

        # Rebuild stencil since subdom sizes have changed. Also may use non-empty AABBs to create halo cells
        self.sim.add_statement(BuildCellListsStencil(self.sim, self.sim.cell_lists)) 

        # Populate cells with local and ghost particles
        self.sim.add_statement(self.sim.update_cells_procedures)   

        # Exchange is not needed all locals are contained in thier owners after deserialization
        # But ghosts must be recreated after rebalancing (optionally uses the halo cells)
        self.sim.add_statement(Borders(self.sim._comm))

        # Reset volatile includes the new locals
        self.sim.add_statement(ResetVolatileProperties(self.sim))  

    @pairs_interface_block
    def reneighbor(self):
        self.sim.module_name('reneighbor')

        reneighboring_procedures = [
            Exchange(self.sim._comm),
            # Note: DomainUpdateLocal must happen after exchange since local particles must be contained in AABBs.
            #       And it must happen before Borders since newly received particles need to be included, so they become ghosts
            #       for their previous neighbor
            DomainUpdateLocal(self.sim),    
            Borders(self.sim._comm),
            BuildCellListsStencil(self.sim, self.sim.cell_lists),
            self.sim.update_cells_procedures,
            ResetVolatileProperties(self.sim)
        ]

        if self.sim._use_contact_history:
            reneighboring_procedures += [
                BuildContactHistory(self.sim, self.sim._contact_history, self.sim.cell_lists),
                ResetContactHistoryUsageStatus(self.sim, self.sim._contact_history),
                ClearUnusedContactHistory(self.sim, self.sim._contact_history)
            ]
        
        self.sim.add_statement(Block.from_list(self.sim, reneighboring_procedures))

    @pairs_interface_block
    def refreshGhosts(self):
        self.sim.module_name('refreshGhosts')
        self.sim.add_statement(Synchronize(self.sim._comm))

    @pairs_interface_block
    def reverseCommunicate(self):
        self.sim.module_name('reverseCommunicate')
        self.sim.add_statement(ReverseComm(self.sim._comm, reduce=True))
    
    @pairs_interface_block
    def resetVolatiles(self):
        self.sim.module_name('resetVolatiles')
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
