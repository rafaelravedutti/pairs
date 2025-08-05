from pairs.ir.assign import Assign
from pairs.ir.branches import Filter
from pairs.ir.loops import For, Continue, Break
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
from pairs.ir.print import Print
from pairs.ir.cast import Cast
from pairs.ir.math import Abs

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

    def halo_condition(self, x, y, z, spacing, layers):
        raise Exception("Regular6DStencil does not support halo cells yet.")
    
    def update_neighborhood(self):
        Assign(self.sim, self.rank, Call_Int(self.sim, "pairs_runtime->getDomainPartitioner()->getRank", []))

        Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['neighbor_ranks', self.neighbor_ranks, self.sim.ndims() * 2])
        Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['pbc', self.pbc, self.sim.ndims() * 2])
        Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['subdom', self.subdom, self.sim.ndims() * 2])

        if isinstance(self.sim.grid, MutableGrid):
            for d in range(self.sim.dims):
                Assign(self.sim, self.sim.grid.min(d), Call(self.sim, "pairs_runtime->getDomainPartitioner()->getMin", [d], Types.Real))
                Assign(self.sim, self.sim.grid.max(d), Call(self.sim, "pairs_runtime->getDomainPartitioner()->getMax", [d], Types.Real))

    def rebalance(self):
        pass

    def update_local(self):
        pass

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

    #### cell lists need to be rebuilt in each phase
    #### --------------------------------------------------------------------
    # def ghost_particles_halo_cells(self, step, position, offset=0.0):
    #     # Particles with one of the following flags are ignored
    #     flags_to_exclude = (Flags.Infinite | Flags.Global)

    #     def next_neighbor(self, j, step, position, offset, flags_to_exclude):
    #         particle_flags = self.sim.particle_flags
    #         cells_to_check = self.sim.cell_lists.halo_cells
    #         ncells_to_check = self.sim.cell_lists.halo_ncells
    #         for nc in For(self.sim, 0, ncells_to_check):
    #             c = self.sim.add_temp_var(0)
    #             Assign(self.sim, c, cells_to_check[nc])
    #             for p in For(self.sim, 0, self.sim.cell_lists.cell_sizes[c]):
    #                 i = self.sim.cell_lists.cell_particles[c][p]
    #                 particle_flags = self.sim.particle_flags

    #                 # Don't check ghost particles if in Exchange mode
    #                 if offset==0.0:
    #                     for _ in Filter(self.sim, i >= self.sim.nlocal):
    #                         Continue(self.sim)()

    #                 for _ in Filter(self.sim, ScalarOp.cmp(particle_flags[i] & flags_to_exclude, 0)):
    #                     for _ in Filter(self.sim, position[i][step] < self.subdom[j] + offset):
    #                         pbc_shifts = [0 if d != step else self.pbc[j] for d in range(self.sim.ndims())]
    #                         yield i, j, self.neighbor_ranks[j], pbc_shifts

    #     def prev_neighbor(self, j, step, position, offset, flags_to_exclude):
    #         particle_flags = self.sim.particle_flags
    #         cells_to_check = self.sim.cell_lists.halo_cells
    #         ncells_to_check = self.sim.cell_lists.halo_ncells
    #         for nc in For(self.sim, 0, ncells_to_check):
    #             c = self.sim.add_temp_var(0)
    #             Assign(self.sim, c, cells_to_check[nc])
    #             for p in For(self.sim, 0, self.sim.cell_lists.cell_sizes[c]):
    #                 i = self.sim.cell_lists.cell_particles[c][p]
    #                 particle_flags = self.sim.particle_flags
                    
    #                 # Don't check ghost particles if in Exchange mode
    #                 if offset==0.0:
    #                     for _ in Filter(self.sim, i >= self.sim.nlocal):
    #                         Continue(self.sim)()
                            
    #                 for _ in Filter(self.sim, ScalarOp.cmp(particle_flags[i] & flags_to_exclude, 0)):
    #                     for _ in Filter(self.sim, position[i][step] > self.subdom[j] - offset):
    #                         pbc_shifts = [0 if d != step else self.pbc[j] for d in range(self.sim.ndims())]
    #                         yield i, j, self.neighbor_ranks[j], pbc_shifts

    #     if self.sim._pbc[step]:
    #         yield from next_neighbor(self, step * 2 + 0, step, position, offset, flags_to_exclude)
    #         yield from prev_neighbor(self, step * 2 + 1, step, position, offset, flags_to_exclude)

    #     else:
    #         j = step * 2 + 0
    #         for _ in Filter(self.sim, ScalarOp.inline(ScalarOp.cmp(self.pbc[j], 0))):
    #             yield from next_neighbor(self, j, step, position, offset, flags_to_exclude)

    #         j = step * 2 + 1
    #         for _ in Filter(self.sim, ScalarOp.inline(ScalarOp.cmp(self.pbc[j], 0))):
    #             yield from prev_neighbor(self, j, step, position, offset, flags_to_exclude)

class BlockForest:
    def __init__(self, sim):
        self.sim                    = sim
        self.reduce_step            = sim.add_var('reduce_step', Types.Int32)   # this var is treated as a tmp (workaround for gpu)
        self.reduce_step.force_read = True
        self.rank                   = sim.add_var('rank', Types.Int32)
        self.nranks                 = sim.add_var('nranks', Types.Int32)
        self.nranks_capacity        = sim.add_var('nranks_capacity', Types.Int32, init_value=27)
        self.total_num_neigh_aabbs  = sim.add_var('total_num_neigh_aabbs', Types.Int32)
        self.num_local_aabbs        = sim.add_var('num_local_aabbs', Types.Int32)
        self.neigh_aabb_capacity    = sim.add_var('neigh_aabb_capacity', Types.Int32, init_value=27)
        self.local_aabb_capacity    = sim.add_var('local_aabb_capacity', Types.Int32, init_value=1)
        self.ranks                  = sim.add_array('ranks', [self.nranks_capacity], Types.Int32)
        self.num_neigh_aabbs        = sim.add_array('num_neigh_aabbs', [self.nranks_capacity], Types.Int32)
        self.aabb_offsets           = sim.add_array('aabb_offsets', [self.nranks_capacity], Types.Int32)
        self.neigh_aabbs            = sim.add_array('neigh_aabbs', [self.neigh_aabb_capacity, 6], Types.Real)
        self.local_aabbs            = sim.add_array('local_aabbs', [self.local_aabb_capacity, 6], Types.Real)
        self.non_empty_local_aabbs  = sim.add_array('non_empty_local_aabbs', [self.local_aabb_capacity], Types.Int32)
        self.subdom                 = sim.add_array('subdom', [sim.ndims() * 2], Types.Real)
        self.has_non_empty_aabb_in_neighborhood_of_rank = sim.add_array('has_non_empty_aabb_in_neighborhood_of_rank', [self.nranks_capacity], Types.Int32)
    
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
    
    def halo_condition(self, x, y, z, spacing, layers):
        # [Experimental option] Can reduce the number of halo cells generated, but comes with overhead
        optimize_paddings = self.sim._optimize_halo_paddings

        for aabb_id in For(self.sim, 0, self.num_local_aabbs):
            for _ in Filter(self.sim, self.non_empty_local_aabbs[aabb_id]):
                aabb = [self.local_aabbs[aabb_id][i] for i in range(6)]

                # Index meaning: halo_[dim][dim_min/dim_max][min/max for each padding]
                if optimize_paddings:
                    tol = 1e-6

                    # Padding X
                    #---------------------------------------------------------------------
                    float_00 = (aabb[0] - (self.min(0) - spacing[0])) / spacing[0]
                    int_00 = Cast.int(self.sim, float_00)
                    sub_00 = tol < Abs(self.sim, int_00 - float_00 + layers[0])
                    add_00 = tol < Abs(self.sim, int_00 - float_00)
                    halo_000 = Select(self.sim, sub_00, int_00 - layers[0], int_00)
                    halo_001 = Select(self.sim, add_00, int_00 + layers[0], int_00)


                    float_01 = (aabb[1] - (self.min(0) - spacing[0])) / spacing[0]
                    int_01 = Cast.int(self.sim, float_01)
                    sub_01 = tol < Abs(self.sim, int_01 - float_01 + layers[0])
                    add_01 = tol < Abs(self.sim, int_01 - float_01)
                    halo_010 = Select(self.sim, sub_01, int_01 - layers[0], int_01)
                    halo_011 = Select(self.sim, add_01, int_01 + layers[0], int_01)

                    # Padding Y
                    #---------------------------------------------------------------------
                    float_10 = (aabb[2] - (self.min(1) - spacing[1])) / spacing[1]
                    int_10 = Cast.int(self.sim, float_10)
                    sub_10 = tol < Abs(self.sim, int_10 - float_10 + layers[1])
                    add_10 = tol < Abs(self.sim, int_10 - float_10)
                    halo_100 = Select(self.sim, sub_10, int_10 - layers[1], int_10)
                    halo_101 = Select(self.sim, add_10, int_10 + layers[1], int_10)

                    float_11 = (aabb[3] - (self.min(1) - spacing[1])) / spacing[1]
                    int_11 = Cast.int(self.sim, float_11)
                    sub_11 = tol < Abs(self.sim, int_11 - float_11 + layers[1])
                    add_11 = tol < Abs(self.sim, int_11 - float_11)
                    halo_110 = Select(self.sim, sub_11, int_11 - layers[1], int_11)
                    halo_111 = Select(self.sim, add_11, int_11 + layers[1], int_11)

                    # Padding Z
                    #---------------------------------------------------------------------
                    float_20 = (aabb[4] - (self.min(2) - spacing[2])) / spacing[2]
                    int_20 = Cast.int(self.sim, float_20)
                    sub_20 = tol < Abs(self.sim, int_20 - float_20 + layers[2])
                    add_20 = tol < Abs(self.sim, int_20 - float_20)
                    halo_200 = Select(self.sim, sub_20, int_20 - layers[2], int_20)
                    halo_201 = Select(self.sim, add_20, int_20 + layers[2], int_20)

                    float_21 = (aabb[5] - (self.min(2) - spacing[2])) / spacing[2]
                    int_21 = Cast.int(self.sim, float_21)
                    sub_21 = tol < Abs(self.sim, int_21 - float_21 + layers[2])
                    add_21 = tol < Abs(self.sim, int_21 - float_21)
                    halo_210 = Select(self.sim, sub_21, int_21 - layers[2], int_21)
                    halo_211 = Select(self.sim, add_21, int_21 + layers[2], int_21)

                else:
                    halo_000 = Cast.int(self.sim, (aabb[0] - (self.min(0) - spacing[0])) / spacing[0]) - layers[0]
                    halo_001 = Cast.int(self.sim, (aabb[0] - (self.min(0) - spacing[0])) / spacing[0]) + layers[0]
                    halo_010 = Cast.int(self.sim, (aabb[1] - (self.min(0) - spacing[0])) / spacing[0]) - layers[0]
                    halo_011 = Cast.int(self.sim, (aabb[1] - (self.min(0) - spacing[0])) / spacing[0]) + layers[0]
                    
                    halo_100 = Cast.int(self.sim, (aabb[2] - (self.min(1) - spacing[1])) / spacing[1]) - layers[1]
                    halo_101 = Cast.int(self.sim, (aabb[2] - (self.min(1) - spacing[1])) / spacing[1]) + layers[1]
                    halo_110 = Cast.int(self.sim, (aabb[3] - (self.min(1) - spacing[1])) / spacing[1]) - layers[1]
                    halo_111 = Cast.int(self.sim, (aabb[3] - (self.min(1) - spacing[1])) / spacing[1]) + layers[1]
                    
                    halo_200 = Cast.int(self.sim, (aabb[4] - (self.min(2) - spacing[2])) / spacing[2]) - layers[2]
                    halo_201 = Cast.int(self.sim, (aabb[4] - (self.min(2) - spacing[2])) / spacing[2]) + layers[2]
                    halo_210 = Cast.int(self.sim, (aabb[5] - (self.min(2) - spacing[2])) / spacing[2]) - layers[2]
                    halo_211 = Cast.int(self.sim, (aabb[5] - (self.min(2) - spacing[2])) / spacing[2]) + layers[2]

                c0 = ScalarOp.and_op(x >= halo_000, x <= halo_011)
                c1 = ScalarOp.and_op(y >= halo_100, y <= halo_111)
                c2 = ScalarOp.and_op(z >= halo_200, z <= halo_211)
                
                cell_is_within_padded_aabb = ScalarOp.and_op(ScalarOp.and_op(c0, c1), c2)

                for _ in Filter(self.sim, cell_is_within_padded_aabb):
                    cond_0 = ScalarOp.or_op(x >= halo_010, x <= halo_001)
                    cond_1 = ScalarOp.or_op(y >= halo_110, y <= halo_101)
                    cond_2 = ScalarOp.or_op(z >= halo_210, z <= halo_201)
                    yield ScalarOp.or_op(ScalarOp.or_op(cond_0, cond_1), cond_2) 

    def update_local(self):
        Call_Void(self.sim, "pairs_runtime->getDomainPartitioner()->updateLocal", [])
        Assign(self.sim, self.num_local_aabbs, Call_Int(self.sim, "pairs_runtime->getDomainPartitioner()->getNumberOfLocalAABBs", []))
        
        for _ in Filter(self.sim, self.local_aabb_capacity < self.num_local_aabbs):
            Assign(self.sim, self.local_aabb_capacity, self.num_local_aabbs + 4)
            for arr in self.local_aabb_capacity.bonded_arrays():
                    arr.realloc()

        for _ in Filter(self.sim, ScalarOp.neq(self.nranks, 0)):
            if self.sim._target.is_gpu():
                CopyArray(self.sim, self.local_aabbs, Contexts.Host, Actions.WriteOnly, self.num_local_aabbs * 6)
                CopyArray(self.sim, self.non_empty_local_aabbs, Contexts.Host, Actions.WriteOnly, self.num_local_aabbs)
                CopyArray(self.sim, self.has_non_empty_aabb_in_neighborhood_of_rank, Contexts.Host, Actions.WriteOnly, self.nranks)

            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['local_aabbs', self.local_aabbs, self.num_local_aabbs * 6])
            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['non_empty_local_aabbs', self.non_empty_local_aabbs, self.num_local_aabbs])
            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['has_non_empty_aabb_in_neighborhood_of_rank', self.has_non_empty_aabb_in_neighborhood_of_rank, self.nranks])

        if self.sim._target.is_gpu():
            CopyArray(self.sim, self.subdom, Contexts.Host, Actions.WriteOnly)

        Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['subdom', self.subdom, self.sim.ndims() * 2])
        if isinstance(self.sim.grid, MutableGrid):
            for d in range(self.sim.dims):
                Assign(self.sim, self.sim.grid.min(d), Call(self.sim, "pairs_runtime->getDomainPartitioner()->getMin", [d], Types.Real))
                Assign(self.sim, self.sim.grid.max(d), Call(self.sim, "pairs_runtime->getDomainPartitioner()->getMax", [d], Types.Real))

    def update_neighborhood(self):
        Call_Void(self.sim, "pairs_runtime->getDomainPartitioner()->updateNeighborhood", [])
        Assign(self.sim, self.rank, Call_Int(self.sim, "pairs_runtime->getDomainPartitioner()->getRank", []))
        Assign(self.sim, self.nranks, Call_Int(self.sim, "pairs_runtime->getNumberOfNeighborRanks", []))
        Assign(self.sim, self.total_num_neigh_aabbs, Call_Int(self.sim, "pairs_runtime->getNumberOfNeighborAABBs", []))

        for _ in Filter(self.sim, ScalarOp.neq(self.nranks, 0)):
            for _ in Filter(self.sim, self.nranks_capacity < self.nranks):
                Assign(self.sim, self.nranks_capacity, self.nranks + 10)
                for arr in self.nranks_capacity.bonded_arrays():
                    arr.realloc()

            for _ in Filter(self.sim, self.neigh_aabb_capacity < self.total_num_neigh_aabbs):
                Assign(self.sim, self.neigh_aabb_capacity, self.total_num_neigh_aabbs + 20)
                for arr in self.neigh_aabb_capacity.bonded_arrays():
                    arr.realloc()

            if self.sim._target.is_gpu():
                CopyArray(self.sim, self.ranks, Contexts.Host, Actions.WriteOnly, self.nranks)
                CopyArray(self.sim, self.num_neigh_aabbs, Contexts.Host, Actions.WriteOnly, self.nranks)
                CopyArray(self.sim, self.aabb_offsets, Contexts.Host, Actions.WriteOnly, self.nranks)
                CopyArray(self.sim, self.neigh_aabbs, Contexts.Host, Actions.WriteOnly, self.total_num_neigh_aabbs * 6)

            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['ranks', self.ranks, self.nranks])
            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['num_neigh_aabbs', self.num_neigh_aabbs, self.nranks])
            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['aabb_offsets', self.aabb_offsets, self.nranks])
            Call_Void(self.sim, "pairs_runtime->copyRuntimeArray", ['neigh_aabbs', self.neigh_aabbs, self.total_num_neigh_aabbs * 6])
            
    def rebalance(self):
        Call_Void(self.sim, "pairs_runtime->getDomainPartitioner()->rebalance", [])
        self.update_neighborhood()

    def ghost_particles(self, step, position, offset=0.0):
        if self.sim._use_halo_cells:
            yield from self.ghost_particles_halo_cells(step, position, offset)
        else:
            yield from self.ghost_particles_original(step, position, offset)

    def ghost_particles_original(self, step, position, offset=0.0):
        ''' TODO :  If we have pbc, a sinlge particle can be a ghost particle multiple times (at different locations) for the same neighbor block,
                    so this function should have the capability to yield more than one particle for every neighbor.
                    But currently it doesn't have that capability, so we need at least 2 blocks in the dimensions that we have pbc.
                    (eg: a particle in a 1x1x1 block config with pbc <ture, true, true> can be ghost at 7 other locations)
        '''
        # Particles with one of the following flags are ignored
        flags_to_exclude = (Flags.Infinite | Flags.Global)
        early_skipping = False

        for r in self.step_indexes(0):     # for every neighbor rank
            for _ in Filter(self.sim, self.has_non_empty_aabb_in_neighborhood_of_rank[r]):
                for i in For(self.sim, 0, self.sim.nlocal):     # for every local particle in this rank
                    particle_flags = self.sim.particle_flags

                    for _ in Filter(self.sim, ScalarOp.cmp(particle_flags[i] & flags_to_exclude, 0)):

                        # ---------------------------------------------------------------------
                        # [EXPERIMENTAL]
                        # If particle is inside one of the previously determined non-empty local blocks, it gets skipped.
                        # else it gets checked with all neighbors
                        # Extreme case in regular partitioning: 
                        # (Particle checks position in one local aabb and avoids 26 neighbor checks)
                        if early_skipping:
                            skip_particle = self.sim.add_temp_var(0)
                            for aabb_id in For(self.sim, 0, self.num_local_aabbs):
                                for _ in Filter(self.sim, self.non_empty_local_aabbs[aabb_id]):
                                    full_cond = None
                                    for d in range(self.sim.ndims()):
                                        aabb_min = self.local_aabbs[aabb_id][d * 2 + 0]
                                        aabb_max = self.local_aabbs[aabb_id][d * 2 + 1]
                                        pos = position[i][d]
                                        d_cond = ScalarOp.and_op(pos >= aabb_min + offset, pos < aabb_max - offset)
                                        full_cond = d_cond if full_cond is None else ScalarOp.and_op(full_cond, d_cond)
                                    
                                    for _ in Filter(self.sim, full_cond):
                                        Assign(self.sim, skip_particle, 1)
                                        Break(self.sim)()

                            for _ in Filter(self.sim, skip_particle):
                                Continue(self.sim)()                            
                        # #---------------------------------------------------------------------

                        for aabb_id in For(self.sim, self.aabb_offsets[r], self.aabb_offsets[r] + self.num_neigh_aabbs[r]): # for every aabb of this neighbor
                            for _ in Filter(self.sim, ScalarOp.neq(self.ranks[r] , self.rank)):     # if my neighobr is not my own rank
                                full_cond = None
                                pbc_shifts = []

                                for d in range(self.sim.ndims()):
                                    aabb_min = self.neigh_aabbs[aabb_id][d * 2 + 0]
                                    aabb_max = self.neigh_aabbs[aabb_id][d * 2 + 1]
                                    d_pbc = 0
                                    d_length = self.sim.grid.length(d)

                                    if self.sim._pbc[d]:
                                        center = aabb_min + (aabb_max - aabb_min) * 0.5     # center of neighbor block
                                        dist = position[i][d] - center                      # distance of our particle from center of neighbor
                                        cond_pbc_neg = dist >=  (d_length * 0.5)
                                        cond_pbc_pos = dist < -(d_length * 0.5)

                                        d_pbc = Select(self.sim, cond_pbc_neg, -1, Select(self.sim, cond_pbc_pos, 1, 0))

                                    adj_pos = position[i][d] + d_pbc * d_length 
                                    d_cond = ScalarOp.and_op(adj_pos >= aabb_min - offset, adj_pos < aabb_max + offset)
                                    full_cond = d_cond if full_cond is None else ScalarOp.and_op(full_cond, d_cond)
                                    pbc_shifts.append(d_pbc)

                                for _ in Filter(self.sim, full_cond):
                                    yield i, r, self.ranks[r], pbc_shifts
                                    Break(self.sim)()

                            for _ in Filter(self.sim, ScalarOp.cmp(self.ranks[r] , self.rank)):     # if my neighbor is me
                                pbc_shifts = []
                                isghost = self.sim.add_temp_var(0)

                                for d in range(self.sim.ndims()):
                                    aabb_min = self.neigh_aabbs[aabb_id][d * 2 + 0]
                                    aabb_max = self.neigh_aabbs[aabb_id][d * 2 + 1]
                                    center = aabb_min + (aabb_max - aabb_min) * 0.5     # center of neighbor block
                                    dist = position[i][d] - center                      # distance of our particle from center of neighbor
                                    d_pbc = 0
                                    d_length = self.sim.grid.length(d)

                                    if self.sim._pbc[d]:
                                        cond_pbc_neg = dist >=  (d_length*0.5 - offset)
                                        cond_pbc_pos = dist < -(d_length*0.5 - offset)
                                        d_pbc = Select(self.sim, cond_pbc_neg, -1, Select(self.sim, cond_pbc_pos, 1, 0))
                                        isghost = ScalarOp.or_op(isghost, d_pbc)

                                    pbc_shifts.append(d_pbc)
                                
                                for _ in Filter(self.sim, isghost):
                                    yield i, r, self.ranks[r], pbc_shifts
                                    Break(self.sim)()


    def ghost_particles_halo_cells(self, step, position, offset=0.0):
        # Particles with one of the following flags are ignored
        flags_to_exclude = (Flags.Infinite | Flags.Global)
        cells_to_check = self.sim.cell_lists.halo_cells
        ncells_to_check = self.sim.cell_lists.halo_ncells
        for r in self.step_indexes(0):     # for every neighbor rank
            for _ in Filter(self.sim, self.has_non_empty_aabb_in_neighborhood_of_rank[r]):
                for nc in For(self.sim, 0, ncells_to_check):
                    c = self.sim.add_temp_var(0)
                    Assign(self.sim, c, cells_to_check[nc])
                    for p in For(self.sim, 0, self.sim.cell_lists.cell_sizes[c]):
                        i = self.sim.cell_lists.cell_particles[c][p]
                        particle_flags = self.sim.particle_flags

                        # Skip ghost particles
                        for _ in Filter(self.sim, i >= self.sim.nlocal):
                            Continue(self.sim)()

                        for _ in Filter(self.sim, ScalarOp.cmp(particle_flags[i] & flags_to_exclude, 0)):
                            for aabb_id in For(self.sim, self.aabb_offsets[r], self.aabb_offsets[r] + self.num_neigh_aabbs[r]): # for every aabb of this neighbor
                                for _ in Filter(self.sim, ScalarOp.neq(self.ranks[r] , self.rank)):     # if my neighobr is not my own rank
                                    full_cond = None
                                    pbc_shifts = []

                                    for d in range(self.sim.ndims()):
                                        aabb_min = self.neigh_aabbs[aabb_id][d * 2 + 0]
                                        aabb_max = self.neigh_aabbs[aabb_id][d * 2 + 1]
                                        d_pbc = 0
                                        d_length = self.sim.grid.length(d)

                                        if self.sim._pbc[d]:
                                            center = aabb_min + (aabb_max - aabb_min) * 0.5     # center of neighbor block
                                            dist = position[i][d] - center                      # distance of our particle from center of neighbor
                                            cond_pbc_neg = dist >=  (d_length * 0.5)
                                            cond_pbc_pos = dist < -(d_length * 0.5)

                                            d_pbc = Select(self.sim, cond_pbc_neg, -1, Select(self.sim, cond_pbc_pos, 1, 0))

                                        adj_pos = position[i][d] + d_pbc * d_length 
                                        d_cond = ScalarOp.and_op(adj_pos >= aabb_min - offset, adj_pos < aabb_max + offset)
                                        full_cond = d_cond if full_cond is None else ScalarOp.and_op(full_cond, d_cond)
                                        pbc_shifts.append(d_pbc)

                                    for _ in Filter(self.sim, full_cond):
                                        yield i, r, self.ranks[r], pbc_shifts
                                        Break(self.sim)()

                                for _ in Filter(self.sim, ScalarOp.cmp(self.ranks[r] , self.rank)):     # if my neighbor is me
                                    pbc_shifts = []
                                    isghost = self.sim.add_temp_var(0)

                                    for d in range(self.sim.ndims()):
                                        aabb_min = self.neigh_aabbs[aabb_id][d * 2 + 0]
                                        aabb_max = self.neigh_aabbs[aabb_id][d * 2 + 1]
                                        center = aabb_min + (aabb_max - aabb_min) * 0.5     # center of neighbor block
                                        dist = position[i][d] - center                      # distance of our particle from center of neighbor
                                        d_pbc = 0
                                        d_length = self.sim.grid.length(d)

                                        if self.sim._pbc[d]:
                                            cond_pbc_neg = dist >=  (d_length*0.5 - offset)
                                            cond_pbc_pos = dist < -(d_length*0.5 - offset)
                                            d_pbc = Select(self.sim, cond_pbc_neg, -1, Select(self.sim, cond_pbc_pos, 1, 0))
                                            isghost = ScalarOp.or_op(isghost, d_pbc)

                                        pbc_shifts.append(d_pbc)
                                    
                                    for _ in Filter(self.sim, isghost):
                                        yield i, r, self.ranks[r], pbc_shifts
                                        Break(self.sim)()
