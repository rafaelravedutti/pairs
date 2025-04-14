
from pairs.ir.assign import Assign
from pairs.ir.scalars import ScalarOp
from pairs.ir.block import pairs_inline, pairs_device_block, pairs_host_block
from pairs.ir.branches import Filter
from pairs.ir.loops import For, ParticleFor
from pairs.ir.types import Types
from pairs.ir.device import CopyArray
from pairs.ir.contexts import Contexts
from pairs.ir.actions import Actions
from pairs.ir.sizeof import Sizeof
from pairs.ir.functions import Call_Void
from pairs.ir.cast import Cast
from pairs.sim.flags import Flags
from pairs.sim.lowerable import Lowerable
from pairs.sim.interaction import ParticleInteraction


class GlobalLocalInteraction(ParticleInteraction):
    def __init__(self, sim, module_name, nbody, cutoff_radius=None, use_cell_lists=False):
        super().__init__(sim, module_name, nbody, cutoff_radius, use_cell_lists)

    @pairs_device_block
    def lower(self):
        self.sim.module_name(f"{self.module_name}_global_local_interactions")
        for ishape in range(self.maxs): # shape of globals
            if self.include_shape(ishape):
                for jshape in range(self.maxs): # shape of locals
                    if self.include_interaction(ishape, jshape):
                        # Globals are presenet in all ranks so they should not interact with ghosts
                        for j in ParticleFor(self.sim):
                            # Loop over the global cell
                            for p in For(self.sim, 0, self.cell_lists.cell_sizes[0]):
                                i = self.cell_lists.cell_particles[0][p]
                                # TODO: Skip if the bounding box of the global body doesn't intersect the subdom of this rank
                                for _ in Filter(self.sim, ScalarOp.and_op(
                                    ScalarOp.cmp(self.sim.particle_shape[i], self.sim.get_shape_id(ishape)),
                                    self.sim.particle_flags[i] & (Flags.Infinite | Flags.Global))):
                                    # Here we make make sure not to interact with other global bodies, otherwise
                                    # their contributions will get reduced again over all ranks
                                    for _ in Filter(self.sim, ScalarOp.and_op(
                                        ScalarOp.cmp(self.sim.particle_shape[j], self.sim.get_shape_id(jshape)),
                                        ScalarOp.not_op(self.sim.particle_flags[j] & (Flags.Infinite | Flags.Global)))):
                                        for _ in Filter(self.sim, ScalarOp.neq(i, j)):
                                            self.compute_interaction(i, j, ishape, jshape, atomic=True)


class GlobalGlobalInteraction(ParticleInteraction):
    def __init__(self, sim, module_name, nbody, cutoff_radius=None, use_cell_lists=False):
        super().__init__(sim, module_name, nbody, cutoff_radius, use_cell_lists)

    @pairs_device_block
    def lower(self):
        self.sim.module_name(f"{self.module_name}_global_global_interactions")
        if self.sim._target.is_gpu():
            first_cell_bytes = self.sim.add_temp_var(0)
            Assign(self.sim, first_cell_bytes, self.cell_lists.cell_capacity * Sizeof(self.sim, Types.Int32))
            CopyArray(self.sim, self.cell_lists.cell_sizes, Contexts.Host, Actions.ReadOnly, first_cell_bytes)
        
        for ishape in range(self.maxs):
            if self.include_shape(ishape):
                # Loop over the global cell
                for p in For(self.sim, 0, self.cell_lists.cell_sizes[0]):
                    i = self.cell_lists.cell_particles[0][p]
                    for _ in Filter(self.sim, ScalarOp.and_op(
                        ScalarOp.cmp(self.sim.particle_shape[i], self.sim.get_shape_id(ishape)),
                        self.sim.particle_flags[i] & (Flags.Infinite | Flags.Global))):
                        for jshape in range(self.maxs):
                            if self.include_interaction(ishape, jshape):
                                # Loop over the global cell
                                for q in For(self.sim, 0, self.cell_lists.cell_sizes[0]):
                                    j = self.cell_lists.cell_particles[0][q]
                                    # Here we only compute interactions with other global bodies
                                    for _ in Filter(self.sim, ScalarOp.and_op(
                                        ScalarOp.cmp(self.sim.particle_shape[j], self.sim.get_shape_id(jshape)),
                                        (self.sim.particle_flags[j] & (Flags.Infinite | Flags.Global)))):
                                        for _ in Filter(self.sim, ScalarOp.neq(i, j)):
                                            self.compute_interaction(i, j, ishape, jshape)

class GlobalReduction:
    def __init__(self, sim, module_name, particle_interaction):
        self.sim = sim
        self.module_name            = module_name
        self.particle_interaction   = particle_interaction
        self.nglobal_red            = sim.add_var('nglobal_red', Types.Int32)               # Number of global particles that need reduction
        self.nglobal_capacity       = sim.add_var('nglobal_capacity', Types.Int32, 64)
        self.global_elem_capacity   = sim.add_var('global_elem_capacity', Types.Int32, 100)
        self.red_buffer             = sim.add_array('red_buffer', [self.nglobal_capacity, self.global_elem_capacity], Types.Real, arr_sync=False) 
        self.intermediate_buffer    = sim.add_array('intermediate_buffer', [self.nglobal_capacity, self.global_elem_capacity], Types.Real, arr_sync=False)
        self.sorted_idx             = sim.add_array('sorted_idx', [self.nglobal_capacity], Types.Int32, arr_sync=False)
        self.unsorted_idx           = sim.add_array('unsorted_idx', [self.nglobal_capacity], Types.Int32, arr_sync=False)
        self.removed_idx            = sim.add_array('removed_idx', [self.nglobal_capacity], Types.Boolean, arr_sync=False)

        self.red_props = set()
        for ishape in range(self.sim.max_shapes()):
            for jshape in range(self.sim.max_shapes()):
                if self.particle_interaction.include_interaction(ishape, jshape):
                    for app in self.particle_interaction.apply_list[ishape*self.sim.max_shapes() + jshape]:
                        self.red_props.add(app.prop())

    def global_particles(self):
        for p in For(self.sim, 0, self.sim.cell_lists.cell_sizes[0]):
            i = self.sim.cell_lists.cell_particles[0][p]
            for ishape in range(self.sim.max_shapes()):
                if self.particle_interaction.include_shape(ishape):
                    for _ in Filter(self.sim, ScalarOp.and_op(
                        ScalarOp.cmp(self.sim.particle_shape[i], self.sim.get_shape_id(ishape)),
                        self.sim.particle_flags[i] & (Flags.Infinite | Flags.Global))):
                        yield i

    def get_elems_per_particle(self):
        return sum([Types.number_of_elements(self.sim, p.type()) for p in self.red_props])
    

class SortGlobals(Lowerable):
    def __init__(self, global_reduction):
        super().__init__(global_reduction.sim)
        self.global_reduction = global_reduction
        self.sim.add_statement(self)

    @pairs_host_block
    def lower(self):
        self.sim.module_name(f"{self.global_reduction.module_name}_sort_globals")
        nglobal_capacity    = self.global_reduction.nglobal_capacity
        nglobal_red         = self.global_reduction.nglobal_red
        unsorted_idx        = self.global_reduction.unsorted_idx
        sorted_idx          = self.global_reduction.sorted_idx
        removed_idx         = self.global_reduction.removed_idx
        uid                 = self.sim.particle_uid
        self.sim.check_resize(nglobal_capacity, nglobal_red)

        Assign(self.sim, nglobal_red, 0)
        for i in self.global_reduction.global_particles():
            Assign(self.sim, unsorted_idx[nglobal_red], i)
            Assign(self.sim, sorted_idx[nglobal_red], 0)
            Assign(self.sim, removed_idx[nglobal_red], 0)
            Assign(self.sim, nglobal_red, nglobal_red +1)

        min_uid = self.sim.add_temp_var(0, Types.UInt64)
        min_idx = self.sim.add_temp_var(0)

        # Here we sort indices of global bodies with respect to their uid's.
        # The sorted uid's will be in identical order on all ranks. This ensures that the
        # reduced properties are mapped correctly to each global body during inplace reduction.
        for i in For(self.sim, 0, nglobal_red):
            Assign(self.sim, min_uid, -1)   # TODO: Lit max: UINT64_MAX
            Assign(self.sim, min_idx, 0)
            for j in For(self.sim, 0, nglobal_red):
                for _ in Filter(self.sim, ScalarOp.and_op(uid[unsorted_idx[j]] < min_uid,
                                                          ScalarOp.not_op(removed_idx[j]))):
                    Assign(self.sim, min_uid, uid[unsorted_idx[j]])
                    Assign(self.sim, min_idx, j)

            Assign(self.sim, sorted_idx[i], unsorted_idx[min_idx])
            Assign(self.sim, removed_idx[min_idx], 1)


class PackGlobals(Lowerable):
    def __init__(self, global_reduction, save_state=True):
        super().__init__(global_reduction.sim)
        self.global_reduction = global_reduction
        self.save_state = save_state
        self.buffer = global_reduction.intermediate_buffer if save_state else global_reduction.red_buffer
        self.sim.add_statement(self)

    @pairs_device_block
    def lower(self):
        self.sim.module_name(f"{self.global_reduction.module_name}_pack_globals_{'intermediate' if self.save_state else 'reduce'}")
        nglobal_red         = self.global_reduction.nglobal_red
        sorted_idx          = self.global_reduction.sorted_idx
        nelems_per_particle = self.global_reduction.get_elems_per_particle()
        self.buffer.set_stride(1, nelems_per_particle)

        for buffer_idx in For(self.sim, 0, nglobal_red):
            i = sorted_idx[buffer_idx]
            p_offset = 0
            for p in self.global_reduction.red_props:
                if not Types.is_scalar(p.type()):
                    nelems = Types.number_of_elements(self.sim, p.type())
                    for e in range(nelems):
                        Assign(self.sim, self.buffer[buffer_idx][p_offset + e], p[i][e])

                    p_offset += nelems
                else:
                    cast_fn = lambda x: Cast(self.sim, x, Types.Real) if p.type() != Types.Real else x
                    Assign(self.sim, self.buffer[buffer_idx][p_offset], cast_fn(p[i]))
                    p_offset += 1


class ResetReductionProps(Lowerable):
    def __init__(self, global_reduction):
        super().__init__(global_reduction.sim)
        self.global_reduction = global_reduction
        self.sim.add_statement(self)

    @pairs_device_block
    def lower(self):
        self.sim.module_name(f"{self.global_reduction.module_name}_reset_globals")
        nglobal_red         = self.global_reduction.nglobal_red
        sorted_idx          = self.global_reduction.sorted_idx

        for buffer_idx in For(self.sim, 0, nglobal_red):
            i = sorted_idx[buffer_idx]
            for p in self.global_reduction.red_props:
                Assign(self.sim, p[i], 0.0)

class ReduceGlobals(Lowerable):
    def __init__(self, global_reduction):
        super().__init__(global_reduction.sim)
        self.global_reduction = global_reduction
        self.sim.add_statement(self)
        
    @pairs_inline
    def lower(self):
        nelems_total = self.global_reduction.nglobal_red * self.global_reduction.get_elems_per_particle() 
        Call_Void( self.sim, "pairs_runtime->allReduceInplaceSum", [self.global_reduction.red_buffer, nelems_total])


class UnpackGlobals(Lowerable):
    def __init__(self, global_reduction):
        super().__init__(global_reduction.sim)
        self.global_reduction = global_reduction
        self.sim.add_statement(self)

    @pairs_device_block
    def lower(self):
        self.sim.module_name(f"{self.global_reduction.module_name}_unpack_globals")
        nglobal_red = self.global_reduction.nglobal_red
        sorted_idx  = self.global_reduction.sorted_idx
        red_buffer  = self.global_reduction.red_buffer
        intermediate_buffer  = self.global_reduction.intermediate_buffer

        for buffer_idx in For(self.sim, 0, nglobal_red):
            i = sorted_idx[buffer_idx]
            p_offset = 0
            for p in self.global_reduction.red_props:
                if not Types.is_scalar(p.type()):
                    nelems = Types.number_of_elements(self.sim, p.type())
                    for e in range(nelems):
                        Assign(self.sim, p[i][e], red_buffer[buffer_idx][p_offset + e] + intermediate_buffer[buffer_idx][p_offset + e])

                    p_offset += nelems
                else:                    
                    cast_fn = lambda x: Cast(self.sim, x, p.type()) if p.type() != Types.Real else x
                    Assign(self.sim, p[i], cast_fn(red_buffer[buffer_idx][p_offset] + intermediate_buffer[buffer_idx][p_offset + e]))
                    p_offset += 1