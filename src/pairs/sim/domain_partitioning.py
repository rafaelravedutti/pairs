from pairs.ir.assign import Assign
from pairs.ir.branches import Filter
from pairs.ir.loops import For
from pairs.ir.functions import Call_Int, Call_Void
from pairs.ir.scalars import ScalarOp
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.flags import Flags


class DimensionRanges:
    def __init__(self, sim):
        self.sim                = sim
        self.nranks             = 6
        self.nranks_capacity    = self.nranks
        self.neighbor_ranks     = sim.add_static_array('neighbor_ranks', [sim.ndims() * 2], Types.Int32)
        self.pbc                = sim.add_static_array('pbc', [sim.ndims() * 2], Types.Int32)
        self.subdom             = sim.add_static_array('subdom', [sim.ndims() * 2], Types.Real)

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
        Call_Void(self.sim, "pairs->initDomain", [param for delim in grid_array for param in delim])
        Call_Void(self.sim, "pairs->copyRuntimeArray", ['neighbor_ranks', self.neighbor_ranks, self.sim.ndims() * 2])
        Call_Void(self.sim, "pairs->copyRuntimeArray", ['pbc', self.pbc, self.sim.ndims() * 2])
        Call_Void(self.sim, "pairs->copyRuntimeArray", ['subdom', self.subdom, self.sim.ndims() * 2])

    def update(self):
        Call_Void(self.sim, "pairs->updateDomain", [])

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
        self.ntotal_aabbs       = sim.add_var('ntotal_aabbs', Types.Int32)
        self.aabb_capacity      = sim.add_var('aabb_capacity', Types.Int32)
        self.ranks              = sim.add_array('ranks', [self.nranks_capacity], Types.Int32)
        self.naabbs             = sim.add_array('naabbs', [self.nranks_capacity], Types.Int32)
        self.aabb_offsets       = sim.add_array('aabb_offsets', [self.nranks_capacity], Types.Int32)
        self.aabbs              = sim.add_array('aabbs', [self.aabb_capacity, 6], Types.Real)
        self.subdom             = sim.add_array('subdom', [sim.ndims() * 2], Types.Real)

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
        Call_Void(self.sim, "pairs->initDomain", [param for delim in grid_array for param in delim])

    def update(self):
        Call_Void(self.sim, "pairs->updateDomain", [])
        Assign(self.sim, self.nranks, Call_Int(self.sim, "pairs->getNumberOfNeighborRanks", []))
        Assign(self.sim, self.ntotal_aabbs, Call_Int(self.sim, "pairs->getNumberOfNeighborAABBs", []))

        for _ in Filter(self.sim, self.nranks_capacity < self.nranks):
            Assign(self.sim, self.nranks_capacity, self.nranks + 10)
            self.ranks.realloc()
            self.naabbs.realloc()
            self.aabb_offsets.realloc()

        for _ in Filter(self.sim, self.aabb_capacity < self.ntotal_aabbs):
            Assign(self.sim, self.aabb_capacity, self.ntotal_aabbs + 20)
            self.aabbs.realloc()

        Call_Void(self.sim, "pairs->copyRuntimeArray", ['ranks', self.ranks, self.nranks])
        Call_Void(self.sim, "pairs->copyRuntimeArray", ['naabbs', self.naabbs, self.nranks])
        Call_Void(self.sim, "pairs->copyRuntimeArray", ['aabb_offsets', self.aabb_offsets, self.nranks])
        Call_Void(self.sim, "pairs->copyRuntimeArray", ['aabbs', self.aabbs, self.ntotal_aabbs * 6])
        Call_Void(self.sim, "pairs->copyRuntimeArray", ['subdom', self.subdom, self.sim.ndims() * 2])

    def ghost_particles(self, step, position, offset=0.0):
        # Particles with one of the following flags are ignored
        flags_to_exclude = (Flags.Infinite | Flags.Global)

        for r in For(self.sim, 0, self.nranks):
            for i in For(self.sim, 0, self.sim.nlocal):
                particle_flags = self.sim.particle_flags

                for _ in Filter(self.sim, ScalarOp.cmp(particle_flags[i] & flags_to_exclude, 0)):
                    for aabb_id in For(self.sim, self.aabb_offsets[r], self.aabb_offsets[r] + self.naabbs[r]):
                        full_cond = None
                        pbc_shifts = []

                        for d in range(self.sim.ndims()):
                            aabb_min = self.aabbs[aabb_id][d * 2 + 0]
                            aabb_max = self.aabbs[aabb_id][d * 2 + 1]
                            center = aabb_min + (aabb_max - aabb_min) * 0.5
                            dist = position[i][d] - center
                            d_length = self.sim.grid.length(d)

                            cond_pbc_neg = dist >  (d_length * 0.5)
                            cond_pbc_pos = dist < -(d_length * 0.5)
                            d_pbc = Select(self.sim, cond_pbc_neg, -1, Select(self.sim, cond_pbc_pos, 1, 0))

                            adj_pos = position[i][d] + d_pbc * d_length
                            d_cond = ScalarOp.and_op(adj_pos > aabb_min - offset, adj_pos < aabb_max + offset)
                            full_cond = d_cond if full_cond is None else ScalarOp.and_op(full_cond, d_cond)
                            pbc_shifts.append(d_pbc)

                        for _ in Filter(self.sim, full_cond):
                            yield i, r, self.ranks[r], pbc_shifts
