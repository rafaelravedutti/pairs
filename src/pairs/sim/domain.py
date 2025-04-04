from pairs.ir.block import pairs_inline
from pairs.sim.lowerable import Lowerable

class DomainUpdateLocal(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        self.sim.domain_partitioning().update_local()


class DomainUpdateNeighborhood(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        self.sim.domain_partitioning().update_neighborhood()


class DomainRebalance(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        self.sim.domain_partitioning().rebalance()