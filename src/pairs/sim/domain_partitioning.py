from pairs.ir.block import pairs_device_block, pairs_host_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import For, ParticleFor
from pairs.ir.utils import Print
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.lowerable import Lowerable


class DimensionRanges:
    def __init__(self, sim):
        self.sim = sim
        self.neighbor_ranks = sim.add_static_array('neighbor_ranks', [sim.ndims() * 2], Types.Int32)
        self.pbc            = sim.add_static_array('pbc', [sim.ndims() * 2], Types.Int32)
        self.subdom         = sim.add_static_array('subdom', [sim.ndims() * 2], Types.Int32)

    def number_of_steps(self):
        return self.sim.ndims()

    def ghost_particles(self, step, position, offset=0.0):
        for i in For(self.sim, 0, self.sim.nlocal + self.sim.nghost):
            j = step * 2 + 0
            for _ in Filter(self.sim, position[i][step] < self.subdom[j] + offset):
                yield i, self.neighbor_ranks[j], [0 if d != step else self.pbc[j] for d in range(self.sim.ndims())]

        for i in For(self.sim, 0, self.sim.nlocal + self.sim.nghost):
            j = step * 2 + 1
            for _ in Filter(self.sim, position[i][step] > self.subdom[j] - offset):
                yield i, self.neighbor_ranks[j], [0 if d != step else self.pbc[j] for d in range(self.sim.ndims())]
