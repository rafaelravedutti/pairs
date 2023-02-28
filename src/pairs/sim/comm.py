from pairs.ir.atomic import AtomicAdd
from pairs.ir.bin_op import BinOp
from pairs.ir.block import pairs_device_block, pairs_host_block, pairs_inline
from pairs.ir.branches import Branch, Filter
from pairs.ir.cast import Cast
from pairs.ir.functions import Call_Void
from pairs.ir.loops import For, ParticleFor, While
from pairs.ir.utils import Print
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.lowerable import Lowerable


class Comm:
    def __init__(self, sim, dom_part):
        self.sim = sim
        self.dom_part = dom_part
        self.nsend_all      = sim.add_var('nsend_all', Types.Int32)
        self.send_capacity  = sim.add_var('send_capacity', Types.Int32, 100000)
        self.recv_capacity  = sim.add_var('recv_capacity', Types.Int32, 100000)
        self.elem_capacity  = sim.add_var('elem_capacity', Types.Int32, 10)
        self.neigh_capacity = sim.add_var('neigh_capacity', Types.Int32, 6)
        self.nsend          = sim.add_array('nsend', [self.neigh_capacity], Types.Int32)
        self.send_offsets   = sim.add_array('send_offsets', [self.neigh_capacity], Types.Int32)
        self.send_buffer    = sim.add_array('send_buffer', [self.send_capacity, self.elem_capacity], Types.Double)
        self.send_map       = sim.add_array('send_map', [self.send_capacity], Types.Int32)
        self.exchg_flag     = sim.add_array('exchg_flag', [sim.particle_capacity], Types.Int32)
        self.exchg_copy_to  = sim.add_array('exchg_copy_to', [self.send_capacity], Types.Int32)
        self.send_mult      = sim.add_array('send_mult', [self.send_capacity, sim.ndims()], Types.Int32)
        self.nrecv          = sim.add_array('nrecv', [self.neigh_capacity], Types.Int32)
        self.recv_offsets   = sim.add_array('recv_offsets', [self.neigh_capacity], Types.Int32)
        self.recv_buffer    = sim.add_array('recv_buffer', [self.recv_capacity, self.elem_capacity], Types.Double)
        self.recv_map       = sim.add_array('recv_map', [self.recv_capacity], Types.Int32)
        self.recv_mult      = sim.add_array('recv_mult', [self.recv_capacity, sim.ndims()], Types.Int32)

    @pairs_inline
    def synchronize(self):
        prop_list = [self.sim.property(p) for p in ['position']]
        for step in range(self.dom_part.number_of_steps()):
            PackGhostParticles(self, step, prop_list)
            CommunicateData(self, step, prop_list)
            UnpackGhostParticles(self, step, prop_list)

    @pairs_inline
    def borders(self):
        prop_list = [self.sim.property(p) for p in ['mass', 'position']]
        self.nsend_all.set(0)
        self.sim.nghost.set(0)
        for step in range(self.dom_part.number_of_steps()):
            DetermineGhostParticles(self, step, self.sim.cell_spacing())
            CommunicateSizes(self, step)
            SetCommunicationOffsets(self, step)
            PackGhostParticles(self, step, prop_list)
            CommunicateData(self, step, prop_list)
            UnpackGhostParticles(self, step, prop_list)
            self.sim.nghost.add(sum([self.nrecv[j] for j in self.dom_part.step_indexes(step)]))

    @pairs_inline
    def exchange(self):
        prop_list = [self.sim.property(p) for p in ['mass', 'position', 'velocity']]
        for step in range(self.dom_part.number_of_steps()):
            self.nsend_all.set(0)
            self.sim.nghost.set(0)
            for s in range(step):
                for j in self.dom_part.step_indexes(s):
                    self.nsend[j].set(0)
                    self.nrecv[j].set(0)
                    self.send_offsets[j].set(0)
                    self.recv_offsets[j].set(0)

            DetermineGhostParticles(self, step, 0.0)
            CommunicateSizes(self, step)
            SetCommunicationOffsets(self, step)
            PackGhostParticles(self, step, prop_list)
            RemoveExchangedParticles_part1(self)
            RemoveExchangedParticles_part2(self, prop_list)
            CommunicateData(self, step, prop_list)
            UnpackGhostParticles(self, step, prop_list)
            ChangeSizeAfterExchange(self, step)


class CommunicateSizes(Lowerable):
    def __init__(self, comm, step):
        super().__init__(comm.sim)
        self.comm = comm
        self.step = step
        self.sim.add_statement(self)

    @pairs_inline
    def lower(self):
        Call_Void(self.sim, "pairs->communicateSizes", [self.step, self.comm.nsend, self.comm.nrecv])


class CommunicateData(Lowerable):
    def __init__(self, comm, step, prop_list):
        super().__init__(comm.sim)
        self.comm = comm
        self.step = step
        self.prop_list = prop_list
        self.sim.add_statement(self)

    @pairs_inline
    def lower(self):
        elem_size = sum([self.sim.ndims() if p.type() == Types.Vector else 1 for p in self.prop_list])
        Call_Void(self.sim, "pairs->communicateData", [self.step, elem_size,
                                                       self.comm.send_buffer, self.comm.send_offsets, self.comm.nsend,
                                                       self.comm.recv_buffer, self.comm.recv_offsets, self.comm.nrecv])


class DetermineGhostParticles(Lowerable):
    def __init__(self, comm, step, spacing):
        super().__init__(comm.sim)
        self.comm = comm
        self.step = step
        self.spacing = spacing
        self.sim.add_statement(self)

    #@pairs_host_block
    @pairs_device_block
    def lower(self):
        nsend_all = self.comm.nsend_all
        nsend = self.comm.nsend
        nrecv = self.comm.nrecv
        send_map = self.comm.send_map
        send_mult = self.comm.send_mult
        exchg_flag = self.comm.exchg_flag
        is_exchange = (self.spacing == 0.0) # TODO: module_params(self.spacing)
        ghost_or_exchg = "exchange" if is_exchange else "ghost"
        self.sim.module_name(f"determine_{ghost_or_exchg}_particles{self.step}")
        self.sim.check_resize(self.comm.send_capacity, nsend)

        for j in self.comm.dom_part.step_indexes(self.step):
            nsend[j].set(0)
            nrecv[j].set(0)

        if is_exchange:
            for i in ParticleFor(self.sim):
                exchg_flag[i].set(0)

        for i, j, _, pbc in self.comm.dom_part.ghost_particles(self.step, self.sim.position(), self.spacing):
            next_idx = AtomicAdd(self.sim, nsend_all, 1)
            send_map[next_idx].set(i)

            if is_exchange:
                exchg_flag[i].set(1)

            for d in range(self.sim.ndims()):
                send_mult[next_idx][d].set(pbc[d])

            nsend[j].add(1)


class SetCommunicationOffsets(Lowerable):
    def __init__(self, comm, step):
        super().__init__(comm.sim)
        self.comm = comm
        self.step = step
        self.sim.add_statement(self)

    @pairs_host_block
    def lower(self):
        nsend = self.comm.nsend
        nrecv = self.comm.nrecv
        send_offsets = self.comm.send_offsets
        recv_offsets = self.comm.recv_offsets
        self.sim.module_name(f"set_communication_offsets{self.step}")

        isend = 0
        irecv = 0
        for i in range(self.step):
            for j in self.comm.dom_part.step_indexes(i):
                isend += nsend[j]
                irecv += nrecv[j]

        for j in self.comm.dom_part.step_indexes(self.step):
            send_offsets[j].set(isend)
            recv_offsets[j].set(irecv)
            isend += nsend[j]
            irecv += nrecv[j]


class PackGhostParticles(Lowerable):
    def __init__(self, comm, step, prop_list):
        super().__init__(comm.sim)
        self.comm = comm
        self.step = step
        self.prop_list = prop_list
        self.sim.add_statement(self)

    def get_elems_per_particle(self):
        return sum([self.sim.ndims() if p.type() == Types.Vector else 1 for p in self.prop_list])

    #@pairs_host_block
    @pairs_device_block
    def lower(self):
        send_buffer = self.comm.send_buffer
        send_buffer.set_stride(1, self.get_elems_per_particle())
        send_map = self.comm.send_map
        send_mult = self.comm.send_mult
        self.sim.module_name(f"pack_ghost_particles{self.step}_" + "_".join([str(p.id()) for p in self.prop_list]))

        step_indexes = self.comm.dom_part.step_indexes(self.step)
        start = self.comm.send_offsets[step_indexes[0]]
        for i in For(self.sim, start, BinOp.inline(start + sum([self.comm.nsend[j] for j in step_indexes]))):
            p_offset = 0
            m = send_map[i]
            for p in self.prop_list:
                if p.type() == Types.Vector:
                    for d in range(self.sim.ndims()):
                        src = p[m][d]
                        if p == self.sim.position():
                            src += send_mult[i][d] * self.sim.grid.length(d)

                        send_buffer[i][p_offset + d].set(src)

                    p_offset += self.sim.ndims()

                else:
                    cast_fn = lambda x: Cast(self.sim, x, Types.Double) if p.type() != Types.Double else x
                    send_buffer[i][p_offset].set(cast_fn(p[m]))
                    p_offset += 1

            
class UnpackGhostParticles(Lowerable):
    def __init__(self, comm, step, prop_list):
        super().__init__(comm.sim)
        self.comm = comm
        self.step = step
        self.prop_list = prop_list
        self.sim.add_statement(self)

    def get_elems_per_particle(self):
        return sum([self.sim.ndims() if p.type() == Types.Vector else 1 for p in self.prop_list])

    #@pairs_host_block
    @pairs_device_block
    def lower(self):
        nlocal = self.sim.nlocal
        recv_buffer = self.comm.recv_buffer
        recv_buffer.set_stride(1, self.get_elems_per_particle())
        self.sim.module_name(f"unpack_ghost_particles{self.step}_" + "_".join([str(p.id()) for p in self.prop_list]))

        step_indexes = self.comm.dom_part.step_indexes(self.step)
        start = self.comm.recv_offsets[step_indexes[0]]
        for i in For(self.sim, start, BinOp.inline(start + sum([self.comm.nrecv[j] for j in step_indexes]))):
            p_offset = 0
            for p in self.prop_list:
                if p.type() == Types.Vector:
                    for d in range(self.sim.ndims()):
                        p[nlocal + i][d].set(recv_buffer[i][p_offset + d])

                    p_offset += self.sim.ndims()

                else:
                    cast_fn = lambda x: Cast(self.sim, x, p.type()) if p.type() != Types.Double else x
                    p[nlocal + i].set(cast_fn(recv_buffer[i][p_offset]))
                    p_offset += 1


class RemoveExchangedParticles_part1(Lowerable):
    def __init__(self, comm):
        super().__init__(comm.sim)
        self.comm = comm
        self.sim.add_statement(self)

    @pairs_host_block
    def lower(self):
        self.sim.module_name("remove_exchanged_particles_pt1")
        send_pos = self.sim.add_temp_var(self.sim.nlocal - 1)
        for i in For(self.sim, 0, self.comm.nsend_all):
            particle_id = self.comm.send_map[i]
            for need_copy in Branch(self.sim, particle_id < self.sim.nlocal - self.comm.nsend_all):
                if need_copy:
                    for _ in While(self.sim, BinOp.cmp(self.comm.exchg_flag[send_pos], 1)):
                        send_pos.set(send_pos - 1)

                    self.comm.exchg_copy_to[i].set(send_pos)
                    send_pos.set(send_pos - 1)

                else:
                    self.comm.exchg_copy_to[i].set(-1)


class RemoveExchangedParticles_part2(Lowerable):
    def __init__(self, comm, prop_list):
        super().__init__(comm.sim)
        self.comm = comm
        self.prop_list = prop_list
        self.sim.add_statement(self)

    #@pairs_host_block
    @pairs_device_block
    def lower(self):
        self.sim.module_name("remove_exchanged_particles_pt2")
        for i in For(self.sim, 0, self.comm.nsend_all):
            src = self.comm.exchg_copy_to[i]
            for _ in Filter(self.sim, src > 0):
                dst = self.comm.send_map[i]
                for p in self.prop_list:
                    if p.type() == Types.Vector:
                        for d in range(self.sim.ndims()):
                            p[dst][d].set(p[src][d])

                    else:
                        p[dst].set(p[src])

        self.sim.nlocal.set(self.sim.nlocal - self.comm.nsend_all)


class ChangeSizeAfterExchange(Lowerable):
    def __init__(self, comm, step):
        super().__init__(comm.sim)
        self.comm = comm
        self.step = step
        self.sim.add_statement(self)

    @pairs_host_block
    def lower(self):
        sim = self.sim
        sim.module_name(f"change_size_after_exchange{self.step}")
        sim.check_resize(self.sim.particle_capacity, self.sim.nlocal)
        self.sim.nlocal.set(sim.nlocal + sum([self.comm.nrecv[j] for j in self.comm.dom_part.step_indexes(self.step)]))
