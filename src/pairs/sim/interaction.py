from pairs.ir.bin_op import BinOp
from pairs.ir.block import Block, pairs_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import For, ParticleFor
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.sim.lowerable import Lowerable


class NeighborFor():
    def __init__(self, sim, particle, cell_lists, neighbor_lists=None):
        self.sim = sim
        self.particle = particle
        self.cell_lists = cell_lists
        self.neighbor_lists = neighbor_lists

    def __str__(self):
        return f"NeighborFor<{self.particle}>"

    def __iter__(self):
        if self.neighbor_lists is None:
            cl = self.cell_lists
            for s in For(self.sim, 0, cl.nstencil):
                neigh_cell = cl.particle_cell[self.particle] + cl.stencil[s]
                for _ in Filter(self.sim, BinOp.and_op(neigh_cell >= 0, neigh_cell <= cl.ncells)):
                    for nc in For(self.sim, 0, cl.cell_sizes[neigh_cell]):
                        it = cl.cell_particles[neigh_cell][nc]
                        for _ in Filter(self.sim, BinOp.neq(it, self.particle)):
                                yield it
        else:
            neighbor_lists = self.neighbor_lists
            for k in For(self.sim, 0, neighbor_lists.numneighs[self.particle]):
                yield neighbor_lists.neighborlists[self.particle][k]


class ParticleInteraction(Lowerable):
    def __init__(self, sim, nbody, cutoff_radius, block=None, bypass_neighbor_lists=False):
        super().__init__(sim)
        self.nbody = nbody
        self.cutoff_radius = cutoff_radius
        self.bypass_neighbor_lists = bypass_neighbor_lists
        self.i = sim.add_symbol(Types.Int32)
        self.j = sim.add_symbol(Types.Int32)
        self.dp = sim.add_symbol(Types.Vector)
        self.rsq = sim.add_symbol(Types.Double)
        self.block = Block(sim, []) if block is None else block

    def delta(self):
        return self.dp

    def squared_distance(self):
        return self.rsq

    def add_statement(self, stmt):
        self.block.add_statement(stmt)

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter(self)
        yield self.i, self.j
        self.sim.leave()

    @pairs_block
    def lower(self):
        if self.nbody == 2:
            position = self.sim.position()
            cell_lists = self.sim.cell_lists
            neighbor_lists = None if self.bypass_neighbor_lists else self.sim.neighbor_lists
            for i in ParticleFor(self.sim):
                self.i.assign(i)
                for j in NeighborFor(self.sim, i, cell_lists, neighbor_lists):
                    dp = position[i] - position[j]
                    rsq = dp.x() * dp.x() + dp.y() * dp.y() + dp.z() * dp.z()
                    self.j.assign(j)
                    self.dp.assign(dp)
                    self.rsq.assign(rsq)
                    self.sim.add_statement(Filter(self.sim, rsq < self.cutoff_radius, self.block))

        else:
            raise Exception("Interactions among more than two particles is currently not implemented!")
