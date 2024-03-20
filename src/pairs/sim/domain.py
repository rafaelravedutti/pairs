from pairs.ir.block import pairs_inline
from pairs.sim.lowerable import Lowerable


class InitializeDomain(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        self.sim.domain_partitioning().initialize()


class UpdateDomain(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        self.sim.domain_partitioning().update()
