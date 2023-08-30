from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.block import Block, pairs_inline
from pairs.ir.branches import Filter
from pairs.ir.loops import For, ParticleFor
from pairs.ir.types import Types
from pairs.sim.lowerable import Lowerable


class Neighbor(ASTTerm):
    def __init__(self, sim, neighbor_index, cell_id, particle_index):
        super().__init__(sim, ScalarOp)
        self._neighbor_index = neighbor_index
        self._cell_id = cell_id
        self._particle_index = particle_index

    def __str__(self):
        return f"Neighbor<{self._neighbor_index}, {self._cell_id}>"

    def type(self):
        return Types.Int32

    def neighbor_index(self):
        return self._neighbor_index

    def cell_id(self):
        return self._cell_id

    def particle_index(self):
        return self._particle_index


class NeighborFor:
    def number_of_cases(neighbor_lists):
        return 2 if neighbor_lists is None else 1

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
                for _ in Filter(self.sim, ScalarOp.and_op(neigh_cell > 0, neigh_cell < cl.ncells)):
                    for nc in For(self.sim, 0, cl.cell_sizes[neigh_cell]):
                        it = cl.cell_particles[neigh_cell][nc]
                        for _ in Filter(self.sim, ScalarOp.neq(it, self.particle)):
                            yield Neighbor(self.sim, nc, neigh_cell, it)

            # Infinite particles
            for inf_id in For(self.sim, 0, cl.cell_sizes[0]):
                inf_particle = cl.cell_particles[0][inf_id]
                for _ in Filter(self.sim, ScalarOp.neq(inf_particle, self.particle)):
                    yield Neighbor(self.sim, inf_id, 0, inf_particle)

        else:
            neighbor_lists = self.neighbor_lists
            for k in For(self.sim, 0, neighbor_lists.numneighs[self.particle]):
                yield Neighbor(self.sim, k, None, neighbor_lists.neighborlists[self.particle][k])


class ParticleInteraction(Lowerable):
    def __init__(self, sim, nbody, cutoff_radius, bypass_neighbor_lists=False):
        super().__init__(sim)
        self.nbody = nbody
        self.cutoff_radius = cutoff_radius
        self.bypass_neighbor_lists = bypass_neighbor_lists
        self.i = sim.add_symbol(Types.Int32)
        self.dp = sim.add_symbol(Types.Vector)
        self.rsq = sim.add_symbol(Types.Double)
        self.ncases = NeighborFor.number_of_cases(None if bypass_neighbor_lists else sim.neighbor_lists)
        self.jlist = [sim.add_symbol(Types.Int32) for _ in range(self.ncases)]
        self.jblocks = [Block(sim, []) for _ in range(self.ncases)]
        self.current_j = -1

    def delta(self):
        return self.dp

    def squared_distance(self):
        return self.rsq

    def add_statement(self, stmt):
        self.jblocks[self.current_j].add_statement(stmt)

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter(self)

        # Neighbors vary across iterations
        for j in range(self.ncases):
            self.current_j = j
            yield self.i, self.jlist[j]

        self.sim.leave()
        self.current_j = -1

    @pairs_inline
    def lower(self):
        if self.nbody == 2:
            position = self.sim.position()
            cell_lists = self.sim.cell_lists
            neighbor_lists = None if self.bypass_neighbor_lists else self.sim.neighbor_lists
            for i in ParticleFor(self.sim):
                j = 0
                self.i.assign(i)

                for neigh in NeighborFor(self.sim, i, cell_lists, neighbor_lists):
                    dp = position[i] - position[neigh.particle_index()]
                    rsq = dp.x() * dp.x() + dp.y() * dp.y() + dp.z() * dp.z()
                    self.jlist[j].assign(neigh.particle_index())
                    self.dp.assign(dp)
                    self.rsq.assign(rsq)
                    self.sim.add_statement(Filter(self.sim, rsq < self.cutoff_radius, self.jblocks[j]))
                    j += 1

        else:
            raise Exception("Interactions among more than two particles is currently not implemented!")
