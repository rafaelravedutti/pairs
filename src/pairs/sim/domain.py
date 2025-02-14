from pairs.ir.block import pairs_inline
from pairs.ir.parameters import Parameter
from pairs.ir.types import Types
from pairs.ir.assign import Assign
from pairs.sim.lowerable import Lowerable


class InitializeDomain(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        self.sim.domain_partitioning().initialize()

class SetDomain(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        for d in range(self.sim.ndims()):
            dmin = Parameter(self.sim, f'd{d}_min', Types.Real)
            Assign(self.sim, self.sim.grid.min(d), dmin)

        for d in range(self.sim.ndims()):
            dmax = Parameter(self.sim, f'd{d}_max', Types.Real)
            Assign(self.sim, self.sim.grid.max(d), dmax)

        self.sim.domain_partitioning().initialize()

class UpdateDomain(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        self.sim.domain_partitioning().update()
