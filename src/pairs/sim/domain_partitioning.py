from pairs.ir.assign import Assign
from pairs.ir.branches import Filter
from pairs.ir.loops import For
from pairs.ir.functions import Call_Int, Call_Void, Call
from pairs.ir.scalars import ScalarOp
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.flags import Flags
from pairs.ir.lit import Lit
from pairs.sim.grid import MutableGrid
from pairs.ir.device import CopyArray
from pairs.ir.contexts import Contexts
from pairs.ir.actions import Actions
from pairs.sim.load_balancing_algorithms import LoadBalancingAlgorithms
from pairs.ir.print import PrintCode
class DimensionRanges:
    def __init__(self, sim):
        self.sim                = sim
        self.nranks             = 6
        self.nranks_capacity    = self.nranks
        self.neighbor_ranks     = sim.add_static_array('neighbor_ranks', [sim.ndims() * 2], Types.Int32)
        self.pbc                = sim.add_static_array('pbc', [sim.ndims() * 2], Types.Int32)
        self.subdom             = sim.add_static_array('subdom', [sim.ndims() * 2], Types.Real)
        self.rank               = sim.add_var('rank', Types.Int32)

    def min(self, dim):
        return self.subdom[dim * 2 + 0]

    def max(self, dim):
        return self.subdom[dim * 2 + 1]

    def number_of_steps(self):
        return self.sim.ndims()

    def step_indexes(self, step):
        return [step * 2 + 0, step * 2 + 1]

    def first_step_index(self, step):
        return self.step_indexes(step)[0]

    def reduce_sum_all_steps(self, array):
        total_size = sum([len(self.step_indexes(s)) for s in range(self.number_of_steps())])
        return sum([array[i] for i in range(total_size)])

    def reduce_sum_step_indexes(self, step, array):
       return sum([array[i] for i in self.step_indexes(step)])

    def initialize(self):
        grid_array = [(self.sim.grid.min(d), self.sim.grid.max(d)) for d in range(self.sim.ndims())]
        Call_Void(self.sim, "pairs_runtime->initDomain", [param for delim in grid_array for param in delim])

    def update(self):
        Call_Void(self.sim, "pairs_runtime->updateDomain", [])
        Assign(self.sim, self.rank, Call_Int(self.sim, "pairs_runtime->getDomainPartitioner()->getRank", []))

        Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['neighbor_ranks', self.neighbor_ranks, self.sim.ndims() * 2])
        Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['pbc', self.pbc, self.sim.ndims() * 2])
        Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['subdom', self.subdom, self.sim.ndims() * 2])

        if isinstance(self.sim.grid, MutableGrid):
            for d in range(self.sim.dims):
                Assign(self.sim, self.sim.grid.min(d), Call(self.sim, "pairs_runtime->getDomainPartitioner()->getMin", [d], Types.Real))
                Assign(self.sim, self.sim.grid.max(d), Call(self.sim, "pairs_runtime->getDomainPartitioner()->getMax", [d], Types.Real))

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
        self.load_balancer      = None
        self.regrid_min         = None
        self.regrid_max         = None
        self.reduce_step        = sim.add_var('reduce_step', Types.Int32)   # this var is treated as a tmp (workaround for gpu)
        self.reduce_step.force_read = True
        self.rank               = sim.add_var('rank', Types.Int32)
        self.nranks             = sim.add_var('nranks', Types.Int32)
        self.nranks_capacity    = sim.add_var('nranks_capacity', Types.Int32, init_value=27)
        self.ntotal_aabbs       = sim.add_var('ntotal_aabbs', Types.Int32)
        self.aabb_capacity      = sim.add_var('aabb_capacity', Types.Int32, init_value=27)
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
        yield from For(self.sim, 0, self.nranks, not_kernel=True)

    def first_step_index(self, step):
        return 0

    def reduce_sum_all_steps(self, array):
        return self.reduce_sum_step_indexes(0, array)

    def reduce_sum_step_indexes(self, step, array):
        Assign(self.sim, self.reduce_step, 0)
        for i in For(self.sim, 0, self.nranks, not_kernel=True):
            Assign(self.sim, self.reduce_step, ScalarOp.inline( self.reduce_step + array[i]))
            
        return self.reduce_step

    def initialize(self):
        grid_array = [(self.sim.grid.min(d), self.sim.grid.max(d)) for d in range(self.sim.ndims())]

        Call_Void(self.sim, "pairs_runtime->initDomain", 
                  [param for delim in grid_array for param in delim] + 
                  self.sim._pbc + ([True] if self.load_balancer is not None else []))
        
        if self.load_balancer is not None:
            PrintCode(self.sim, "pairs_runtime->getDomainPartitioner()->initWorkloadBalancer"
                      f"({LoadBalancingAlgorithms.c_keyword(self.load_balancer)}, {self.regrid_min}, {self.regrid_max});")

            # Call_Void(self.sim, "pairs_runtime->getDomainPartitioner()->initWorkloadBalancer", 
            #           [self.load_balancer, self.regrid_min, self.regrid_max])

    def update(self):
        Call_Void(self.sim, "pairs_runtime->updateDomain", [])
        Assign(self.sim, self.rank, Call_Int(self.sim, "pairs_runtime->getDomainPartitioner()->getRank", []))
        Assign(self.sim, self.nranks, Call_Int(self.sim, "pairs_runtime->getNumberOfNeighborRanks", []))

        for _ in Filter(self.sim, ScalarOp.neq(self.nranks, 0)):
            Assign(self.sim, self.ntotal_aabbs, Call_Int(self.sim, "pairs_runtime->getNumberOfNeighborAABBs", []))

            for _ in Filter(self.sim, self.nranks_capacity < self.nranks):
                Assign(self.sim, self.nranks_capacity, self.nranks + 10)
                self.ranks.realloc()
                self.naabbs.realloc()
                self.aabb_offsets.realloc()

            for _ in Filter(self.sim, self.aabb_capacity < self.ntotal_aabbs):
                Assign(self.sim, self.aabb_capacity, self.ntotal_aabbs + 20)
                self.aabbs.realloc()
            
            CopyArray(self.sim, self.ranks, Contexts.Host, Actions.WriteOnly, self.nranks)
            CopyArray(self.sim, self.naabbs, Contexts.Host, Actions.WriteOnly, self.nranks)
            CopyArray(self.sim, self.aabb_offsets, Contexts.Host, Actions.WriteOnly, self.nranks)
            CopyArray(self.sim, self.aabbs, Contexts.Host, Actions.WriteOnly, self.ntotal_aabbs * 6)
            CopyArray(self.sim, self.subdom, Contexts.Host, Actions.WriteOnly)

            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['ranks', self.ranks, self.nranks])
            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['naabbs', self.naabbs, self.nranks])
            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['aabb_offsets', self.aabb_offsets, self.nranks])
            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['aabbs', self.aabbs, self.ntotal_aabbs * 6])
            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['subdom', self.subdom, self.sim.ndims() * 2])
        
        if isinstance(self.sim.grid, MutableGrid):
            for d in range(self.sim.dims):
                Assign(self.sim, self.sim.grid.min(d), Call(self.sim, "pairs_runtime->getDomainPartitioner()->getMin", [d], Types.Real))
                Assign(self.sim, self.sim.grid.max(d), Call(self.sim, "pairs_runtime->getDomainPartitioner()->getMax", [d], Types.Real))

    def ghost_particles(self, step, position, offset=0.0):
        ''' TODO :  If we have pbc, a sinlge particle can be a ghost particle multiple times (at different locations) for the same neighbor block,
                    so this function should have the capability to yield more than one particle for every neighbor.
                    But currently it doesn't have that capability, so we need at least 2 blocks in the dimensions that we have pbc.
                    (eg: a particle in a 1x1x1 block config with pbc <ture, true, true> can be ghost at 7 other locations)
        '''
        # Particles with one of the following flags are ignored
        flags_to_exclude = (Flags.Infinite | Flags.Global)

        for r in self.step_indexes(0):     # for every neighbor rank
            for i in For(self.sim, 0, self.sim.nlocal):     # for every local particle in this rank
                particle_flags = self.sim.particle_flags

                for _ in Filter(self.sim, ScalarOp.cmp(particle_flags[i] & flags_to_exclude, 0)):
                    for aabb_id in For(self.sim, self.aabb_offsets[r], self.aabb_offsets[r] + self.naabbs[r]): # for every aabb of this neighbor
                        for _ in Filter(self.sim, ScalarOp.neq(self.ranks[r] , self.rank)):     # if my neighobr is not my own rank
                            full_cond = None
                            pbc_shifts = []

                            for d in range(self.sim.ndims()):
                                aabb_min = self.aabbs[aabb_id][d * 2 + 0]
                                aabb_max = self.aabbs[aabb_id][d * 2 + 1]
                                d_pbc = 0
                                d_length = self.sim.grid.length(d)

                                if self.sim._pbc[d]:
                                    center = aabb_min + (aabb_max - aabb_min) * 0.5     # center of neighbor block
                                    dist = position[i][d] - center                      # distance of our particle from center of neighbor
                                    cond_pbc_neg = dist >  (d_length * 0.5)
                                    cond_pbc_pos = dist < -(d_length * 0.5)

                                    d_pbc = Select(self.sim, cond_pbc_neg, -1, Select(self.sim, cond_pbc_pos, 1, 0))

                                adj_pos = position[i][d] + d_pbc * d_length 
                                d_cond = ScalarOp.and_op(adj_pos > aabb_min - offset, adj_pos < aabb_max + offset)
                                full_cond = d_cond if full_cond is None else ScalarOp.and_op(full_cond, d_cond)
                                pbc_shifts.append(d_pbc)

                            for _ in Filter(self.sim, full_cond):
                                yield i, r, self.ranks[r], pbc_shifts

                        for _ in Filter(self.sim, ScalarOp.cmp(self.ranks[r] , self.rank)):     # if my neighbor is me (cuz I'm the only rank in a dimension that has pbc)
                            pbc_shifts = []
                            isghost = Lit(self.sim, 0)

                            for d in range(self.sim.ndims()):
                                aabb_min = self.aabbs[aabb_id][d * 2 + 0]
                                aabb_max = self.aabbs[aabb_id][d * 2 + 1]
                                center = aabb_min + (aabb_max - aabb_min) * 0.5     # center of neighbor block
                                dist = position[i][d] - center                      # distance of our particle from center of neighbor
                                d_pbc = 0
                                d_length = self.sim.grid.length(d)

                                if self.sim._pbc[d]:
                                    cond_pbc_neg = dist >  (d_length*0.5 - offset)
                                    cond_pbc_pos = dist < -(d_length*0.5 - offset)
                                    d_pbc = Select(self.sim, cond_pbc_neg, -1, Select(self.sim, cond_pbc_pos, 1, 0))
                                    isghost = ScalarOp.or_op(isghost, d_pbc)

                                pbc_shifts.append(d_pbc)
                            
                            for _ in Filter(self.sim, isghost):
                                yield i, r, self.ranks[r], pbc_shifts
