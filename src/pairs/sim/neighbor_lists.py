from pairs.ir.assign import Assign
from pairs.ir.block import pairs_device_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.layouts import Layouts
from pairs.ir.loops import ParticleFor
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.sim.interaction import ParticleInteraction
from pairs.sim.lowerable import Lowerable


class NeighborLists:
    def __init__(self, sim, cell_lists):
        neighbor_layout = Layouts.SoA if sim._target.is_gpu() else Layouts.AoS
        self.sim = sim
        self.cell_lists = cell_lists
        self.neighborlists = self.sim.add_array('neighborlists', [self.sim.particle_capacity, self.sim.neighbor_capacity], Types.Int32, neighbor_layout)
        self.numneighs = self.sim.add_array('numneighs', [self.sim.particle_capacity, self.sim.max_shapes()], Types.Int32)


class BuildNeighborLists(Lowerable):
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
        sim.module_name("build_neighbor_lists")
        sim.check_resize(sim.neighbor_capacity, neighbor_lists.numneighs)

        for i in ParticleFor(sim):
            for shape in range(sim.max_shapes()):
                Assign(sim, neighbor_lists.numneighs[i][shape], 0)

        for interaction_data in ParticleInteraction(sim, 2, cutoff_radius, use_cell_lists=True):
            i = interaction_data.i()
            j = interaction_data.j()
            shape = interaction_data.shape()

            neighs_start = sum([neighbor_lists.numneighs[i][s] for s in range(shape)], 0)
            numneighs = neighbor_lists.numneighs[i][shape]
            Assign(sim, neighbor_lists.neighborlists[i][neighs_start + numneighs], j)
            Assign(sim, neighbor_lists.numneighs[i][shape], numneighs + 1)
