from pairs.ir.block import pairs_device_block, pairs_host_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.functions import Call_Void
from pairs.ir.loops import For, ParticleFor
from pairs.ir.utils import Print
from pairs.ir.scalars import ScalarOp
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.flags import Flags
from pairs.sim.lowerable import Lowerable


class DimensionRanges:
    def __init__(self, sim):
        self.sim            = sim
        self.neighbor_ranks = sim.add_static_array('neighbor_ranks', [sim.ndims() * 2], Types.Int32)
        self.pbc            = sim.add_static_array('pbc', [sim.ndims() * 2], Types.Int32)
        self.subdom         = sim.add_static_array('subdom', [sim.ndims() * 2], Types.Real)

    def min(self, dim):
        return self.subdom[dim * 2 + 0]

    def max(self, dim):
        return self.subdom[dim * 2 + 1]

    def number_of_steps(self):
        return self.sim.ndims()

    def step_indexes(self, step):
        return [step * 2 + 0, step * 2 + 1]

    def initialize(self):
        grid_array = [(self.sim.grid.min(d), self.sim.grid.max(d)) for d in range(self.sim.ndims())]
        Call_Void(self.sim, "pairs->initDomain", [param for delim in grid_array for param in delim]),
        Call_Void(self.sim, "pairs->fillCommunicationArrays", [self.neighbor_ranks, self.pbc, self.subdom])

    def ghost_particles(self, step, position, offset=0.0):
        # Particles with one of the following flags are ignored
        flags_to_exclude = (Flags.Infinite | Flags.Global)

        def next_neighbor(self, j, step, position, offset, flags_to_exclude):
            particle_flags = self.sim.particle_flags
            for i in For(self.sim, 0, self.sim.nlocal + self.sim.nghost):
                for _ in Filter(self.sim, ScalarOp.cmp(particle_flags[i] & flags_to_exclude, 0)):
                    for _ in Filter(self.sim, position[i][step] < self.subdom[j] + offset):
                        pbc_shifts = [0 if d != step else self.pbc[j] for d in range(self.sim.ndims())]
                        yield i, j, self.neighbor_ranks[j], pbc_shifts



        def prev_neighbor(self, j, step, position, offset, flags_to_exclude):
            particle_flags = self.sim.particle_flags
            for i in For(self.sim, 0, self.sim.nlocal + self.sim.nghost):
                for _ in Filter(self.sim, ScalarOp.cmp(particle_flags[i] & flags_to_exclude, 0)):
                    for _ in Filter(self.sim, position[i][step] > self.subdom[j] - offset):
                        pbc_shifts = [0 if d != step else self.pbc[j] for d in range(self.sim.ndims())]
                        yield i, j, self.neighbor_ranks[j], pbc_shifts

        if self.sim._pbc[step]:
            yield from next_neighbor(self, step * 2 + 0, step, position, offset, flags_to_exclude)
            yield from prev_neighbor(self, step * 2 + 1, step, position, offset, flags_to_exclude)

        else:
            j = step * 2 + 0
            for _ in Filter(self.sim, ScalarOp.inline(ScalarOp.cmp(self.pbc[j], 0))):
                yield from next_neighbor(self, j, step, position, offset, flags_to_exclude)

            j = step * 2 + 1
            for _ in Filter(self.sim, ScalarOp.inline(ScalarOp.cmp(self.pbc[j], 0))):
                yield from prev_neighbor(self, j, step, position, offset, flags_to_exclude)


class BlockForest:
    def __init__(self, sim):
        self.sim                = sim
        self.nranks             = sim.add_var('nranks', Types.Int32)
        self.nranks_capacity    = sim.add_var('nranks_capacity', Types.Int32)
        self.aabb_capacity      = sim.add_var('aabb_capacity', Types.Int32)
        self.ranks              = sim.add_static_array('ranks', [self.nranks_capacity], Types.Int32)
        self.naabbs             = sim.add_static_array('naabbs', [self.nranks_capacity], Types.Int32)
        self.offsets            = sim.add_static_array('rank_offsets', [self.nranks_capacity], Types.Int32)
        self.pbc                = sim.add_static_array('pbc', [self.aabb_capacity, 3], Types.Int32)
        self.aabbs              = sim.add_static_array('aabbs', [self.aabb_capacity, 6], Types.Real)
        self.subdom             = sim.add_static_array('subdom', [sim.ndims() * 2], Types.Real)

    def min(self, dim):
        return self.subdom[dim * 2 + 0]

    def max(self, dim):
        return self.subdom[dim * 2 + 1]

    def number_of_steps(self):
        return 1

    def step_indexes(self, step):
        return [step]

    def initialize(self):
        grid_array = [(self.sim.grid.min(d), self.sim.grid.max(d)) for d in range(self.sim.ndims())]
        Call_Void(self.sim, "pairs->initDomain", [param for delim in grid_array for param in delim]),
        Call_Void(self.sim, "pairs->fillCommunicationArrays", [self.ranks, self.pbc, self.subdom])

    def ghost_particles(self, step, position, offset=0.0):
        # Particles with one of the following flags are ignored
        flags_to_exclude = (Flags.Infinite | Flags.Global)

        for i in For(self.sim, 0, self.sim.nlocal):
            particle_flags = self.sim.particle_flags

            for _ in Filter(self.sim, ScalarOp.cmp(particle_flags[i] & flags_to_exclude, 0)):
                for r in For(self.sim, 0, self.nranks):
                    for aabb_id in For(self.sim, self.offsets[r], self.offsets[r] + self.naabbs[r]):
                        full_cond = None

                        for d in range(self.sim.ndims()):
                            d_cond = ScalarOp.and_op(
                                position[i][d] > self.aabbs[aabb_id][d * 2 + 0] + offset,
                                position[i][d] < self.aabbs[aabb_id][d * 2 + 1] - offset)

                            full_cond = d_cond if full_cond is None else \
                                        ScalarOp.and_op(full_cond, d_cond)

                        for _ in Filter(self.sim, full_cond):
                            pbc_shifts = [self.pbc[aabb_id][d] for d in range(self.sim.ndims())]
                            yield i, r, self.ranks[r], pbc_shifts
