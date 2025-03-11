from pairs.ir.block import Block, pairs_interface_block
from pairs.ir.functions import Call_Void, Call, Call_Int
from pairs.ir.parameters import Parameter
from pairs.ir.ret import Return
from pairs.ir.scalars import ScalarOp
from pairs.sim.domain import UpdateDomain
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
from pairs.sim.thermo import ComputeThermo

class InterfaceModules:
    def __init__(self, sim):
        self.sim = sim

    def create_all(self):
        self.initialize()
        self.setup_cells()
        self.update_domain()
        self.update_cells(self.sim.reneighbor_frequency) 
        self.communicate(self.sim.reneighbor_frequency)
        self.reverse_comm() 
        self.reset_volatiles()

        if self.sim._use_contact_history:
            if self.neighbor_lists:
                self.build_contact_history(self.sim.reneighbor_frequency)
            self.reset_contact_history()

        if self.sim._compute_thermo != 0:
            self.compute_thermo(self.sim._compute_thermo)

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
            DeclareVariables(self.sim),
            DeclareArrays(self.sim),
            AllocateProperties(self.sim),
            AllocateContactProperties(self.sim),
            AllocateFeatureProperties(self.sim),
            RegisterTimers(self.sim),
            RegisterMarkers(self.sim)
        ])

        if self.sim._enable_profiler:
            PrintCode(self.sim, "LIKWID_MARKER_INIT;")

        self.sim.add_statement(inits)

    @pairs_interface_block
    def setup_cells(self):
        self.sim.module_name('setup_cells')
        
        if self.sim.cell_lists.runtime_spacing:
            for d in range(self.sim.dims):
                Assign(self.sim, self.sim.cell_lists.spacing[d], Parameter(self.sim, f'cell_spacing_d{d}', Types.Real))

        if self.sim.cell_lists.runtime_cutoff_radius:
            Assign(self.sim, self.sim.cell_lists.cutoff_radius, Parameter(self.sim, 'cutoff_radius', Types.Real))

        # This update assumes all particles have been created exactly in the rank that contains them 
        self.sim.add_statement(UpdateDomain(self.sim))  
        self.sim.add_statement(BuildCellListsStencil(self.sim, self.sim.cell_lists))
        
    @pairs_interface_block
    def update_domain(self):
        self.sim.module_name('update_domain')
        self.sim.add_statement(Exchange(self.sim._comm))    # Local particles must be contained in their owners before domain update
        self.sim.add_statement(UpdateDomain(self.sim))
        # Exchange is not needed after update since all locals are contained in thier owners
        self.sim.add_statement(Borders(self.sim._comm))     # Ghosts must be recreated after update
        self.sim.add_statement(ResetVolatileProperties(self.sim))   # Reset volatile includes the new locals
        self.sim.add_statement(BuildCellListsStencil(self.sim, self.sim.cell_lists))    # Rebuild stencil since subdom sizes have changed
        self.sim.add_statement(self.sim.update_cells_procedures)
        
    @pairs_interface_block
    def reset_volatiles(self):
        self.sim.module_name('reset_volatiles')
        self.sim.add_statement(ResetVolatileProperties(self.sim))
    
    @pairs_interface_block
    def update_cells(self, reneighbor_frequency=1):
        self.sim.module_name('update_cells')
        timestep = Parameter(self.sim, f'timestep', Types.Int32)
        cond = ScalarOp.inline(ScalarOp.or_op(
            ScalarOp.cmp((timestep + 1) % reneighbor_frequency, 0),
            ScalarOp.cmp(timestep, 0)
            ))

        self.sim.add_statement(Filter(self.sim, cond, self.sim.update_cells_procedures))

    @pairs_interface_block
    def communicate(self, reneighbor_frequency=1):
        self.sim.module_name('communicate')
        timestep = Parameter(self.sim, f'timestep', Types.Int32)
        cond = ScalarOp.inline(ScalarOp.or_op(
            ScalarOp.cmp((timestep + 1) % reneighbor_frequency, 0),
            ScalarOp.cmp(timestep, 0)
            ))
        
        exchange = Filter(self.sim, cond, Exchange(self.sim._comm))
        border_sync = Branch(self.sim, cond, 
                             blk_if = Borders(self.sim._comm), 
                             blk_else = Synchronize(self.sim._comm))
        
        self.sim.add_statement(exchange)
        self.sim.add_statement(border_sync)
        
        # TODO: Maybe remove this from here, but volatiles must always be reset after exchange
        self.sim.add_statement(Filter(self.sim, cond, Block(self.sim, ResetVolatileProperties(self.sim))))   

    @pairs_interface_block
    def reverse_comm(self):
        self.sim.module_name('reverse_comm')
        self.sim.add_statement(ReverseComm(self.sim._comm, reduce=True))
    
    @pairs_interface_block
    def build_contact_history(self, reneighbor_frequency=1):
        self.sim.module_name('build_contact_history')
        timestep = Parameter(self.sim, f'timestep', Types.Int32)
        cond = ScalarOp.inline(ScalarOp.or_op(
            ScalarOp.cmp((timestep + 1) % reneighbor_frequency, 0),
            ScalarOp.cmp(timestep, 0)
            ))
        
        self.sim.add_statement(
            Filter(self.sim, cond,
                   BuildContactHistory(self.sim, self.sim._contact_history, self.sim.cell_lists)))

    @pairs_interface_block
    def reset_contact_history(self):
        self.sim.module_name('reset_contact_history')
        self.sim.add_statement(ResetContactHistoryUsageStatus(self.sim, self.sim._contact_history))
        self.sim.add_statement(ClearUnusedContactHistory(self.sim, self.sim._contact_history))

    @pairs_interface_block
    def compute_thermo(self):
        self.sim.module_name('compute_thermo')
        self.sim.add_statement(ComputeThermo(self.sim))

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
        Call_Void(self.sim, "pairs::log_timers", [])
        Call_Void(self.sim, "pairs::print_stats", [self.sim.nlocal, self.sim.nghost])
        PrintCode(self.sim, "delete pobj;")
        PrintCode(self.sim, "delete pairs_runtime;")
