from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Void
from pairs.ir.types import Types
from pairs.sim.lowerable import Lowerable


class InitializeDomain(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        dom_part = self.sim.domain_partitioning()
        grid_array = [(self.sim.grid.min(d), self.sim.grid.max(d)) for d in range(self.sim.ndims())]
        Call_Void(self.sim, "pairs->initDomain", [param for delim in grid_array for param in delim]),
        Call_Void(self.sim, "pairs->fillCommunicationArrays", [dom_part.neighbor_ranks, dom_part.pbc, dom_part.subdom])

