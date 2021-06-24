from ir.branches import Branch, Filter
from ir.data_types import Type_Int
from ir.loops import For, ParticleFor, NeighborFor
from ir.utils import Print
from sim.resize import Resize


class NeighborLists:
    def __init__(self, cell_lists):
        self.sim = cell_lists.sim
        self.cell_lists = cell_lists
        self.capacity = self.sim.add_var('neighborlist_capacity', Type_Int, 32)
        self.neighborlists = self.sim.add_array('neighborlists', [self.sim.particle_capacity, self.capacity], Type_Int)
        self.numneighs = self.sim.add_array('numneighs', self.sim.particle_capacity, Type_Int)


class NeighborListsBuild:
    def __init__(self, neighbor_lists):
        self.neighbor_lists = neighbor_lists

    def lower(self):
        neighbor_lists = self.neighbor_lists
        sim = neighbor_lists.sim
        cell_lists = neighbor_lists.cell_lists
        cutoff_radius = cell_lists.cutoff_radius
        position = sim.property('position')

        sim.clear_block()
        sim.add_statement(Print(sim, "NeighborListsBuild"))
        for resize in Resize(sim, neighbor_lists.capacity):
            for i in ParticleFor(sim):
                neighbor_lists.numneighs[i].set(0)
                for j in NeighborFor(sim, i, cell_lists):
                    # TODO: find a way to not repeat this (already present in particle_pairs)
                    dp = position[i] - position[j]
                    rsq = dp.x() * dp.x() + dp.y() * dp.y() + dp.z() * dp.z()
                    for _ in Filter(sim, rsq < cutoff_radius):
                        numneighs = neighbor_lists.numneighs[i]
                        for cond in Branch(sim, numneighs >= neighbor_lists.capacity):
                            if cond:
                                resize.set(numneighs)
                            else:
                                neighbor_lists.neighborlists[i][numneighs].set(j)
                                neighbor_lists.numneighs[i].set(numneighs + 1)

        return sim.block
