from pairs.ir.assign import Assign
from pairs.ir.block import pairs_device_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import ParticleFor
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.sim.interaction import ParticleInteraction
from pairs.sim.lowerable import Lowerable


class NeighborLists:
    def __init__(self, cell_lists):
        self.sim = cell_lists.sim
        self.cell_lists = cell_lists
        self.neighborlists = self.sim.add_array('neighborlists', [self.sim.particle_capacity, self.sim.neighbor_capacity], Types.Int32)
        self.numneighs = self.sim.add_array('numneighs', self.sim.particle_capacity, Types.Int32)


class NeighborListsBuild(Lowerable):
    def __init__(self, sim, neighbor_lists):
        super().__init__(sim)
        self.neighbor_lists = neighbor_lists

    @pairs_device_block
    def lower(self):
        sim = self.sim
        neighbor_lists = self.neighbor_lists
        cell_lists = neighbor_lists.cell_lists
        cutoff_radius = cell_lists.cutoff_radius
        position = sim.position()
        sim.module_name("neighbor_lists_build")
        sim.check_resize(sim.neighbor_capacity, neighbor_lists.numneighs)

        for i in ParticleFor(sim):
            Assign(self.sim, neighbor_lists.numneighs[i], 0)

        for i, j in ParticleInteraction(sim, 2, cutoff_radius, bypass_neighbor_lists=True):
            numneighs = neighbor_lists.numneighs[i]
            Assign(self.sim, neighbor_lists.neighborlists[i][numneighs], j)
            Assign(self.sim, neighbor_lists.numneighs[i], numneighs + 1)
