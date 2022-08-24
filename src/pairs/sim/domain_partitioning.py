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

    def ghost_particles(self, position, offset=0.0):
        for dim in range(0, self.sim.ndims()):
            nall = self.sim.nlocal + self.sim.comm.nghost
            for i in For(sim, 0, nall):
                j = dim * 2
                for _ in Filter(self.sim, position < self.subdom[j] + offset):
                    yield i, self.neighbor_ranks[j], [0 if d != dim else self.pbc[j] for d in self.sim.ndims()]

            for i in For(sim, 0, nall):
                j = dim * 2 + 1
                for _ in Filter(self.sim, position > self.subdom[j] - offset):
                    yield i, self.neighbor_ranks[j], [0 if d != dim else self.pbc[j] for d in self.sim.ndims()]
