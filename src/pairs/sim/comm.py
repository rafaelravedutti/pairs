from pairs.ir.atomic import AtomicAdd
from pairs.ir.bin_op import BinOp
from pairs.ir.block import pairs_device_block, pairs_host_block, pairs_inline
from pairs.ir.branches import Branch, Filter
from pairs.ir.cast import Cast
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
        self.send_capacity  = sim.add_var('send_capacity', Types.Int32, 100)
        self.recv_capacity  = sim.add_var('recv_capacity', Types.Int32, 100)
        self.elem_capacity  = sim.add_var('elem_capacity', Types.Int32, 10)
        self.neigh_capacity = sim.add_var('neigh_capacity', Types.Int32, 6)
        self.nsend          = sim.add_array('nsend', [self.neigh_capacity], Types.Int32)
        self.send_buffer    = sim.add_array('send_buffer', [self.send_capacity, self.elem_capacity], Types.Double)
        self.send_map       = sim.add_array('send_map', [self.send_capacity], Types.Int32)
        self.exchg_flag     = sim.add_array('exchg_flag', [sim.particle_capacity], Types.Int32)
        self.exchg_copy_to  = sim.add_array('exchg_copy_to', [self.send_capacity], Types.Int32)
        self.send_mult      = sim.add_array('send_mult', [self.send_capacity, sim.ndims()], Types.Int32)
        self.nrecv          = sim.add_array('nrecv', [self.neigh_capacity], Types.Int32)
        self.recv_buffer    = sim.add_array('recv_buffer', [self.recv_capacity, self.elem_capacity], Types.Double)
        self.recv_map       = sim.add_array('recv_map', [self.recv_capacity], Types.Int32)
        self.recv_mult      = sim.add_array('recv_mult', [self.recv_capacity, sim.ndims()], Types.Int32)

    @pairs_inline
    def synchronize(self):
        prop_list = [self.sim.property(p) for p in ['position']]
        PackGhostParticles(self, prop_list)
        CommunicateData(self, prop_list)
        UnpackGhostParticles(self, prop_list)

    @pairs_inline
    def borders(self):
        prop_list = [self.sim.property(p) for p in ['mass', 'position']]
        for d in range(self.sim.ndims()):
            DetermineGhostParticles(self, d, self.sim.cell_spacing())
            PackGhostParticles(self, prop_list)
            CommunicateSizes(self)
            CommunicateData(self, prop_list)
            UnpackGhostParticles(self, prop_list)

    @pairs_inline
    def exchange(self):
        prop_list = [self.sim.property(p) for p in ['mass', 'position', 'velocity']]
        for d in range(self.sim.ndims()):
            DetermineGhostParticles(self, d, 0.0)
            PackGhostParticles(self, prop_list)
            RemoveExchangedParticles_part1(self)
            RemoveExchangedParticles_part2(self, prop_list)
            CommunicateSizes(self)
            CommunicateData(self, prop_list)
            ChangeSizeAfterExchange(self)
            UnpackGhostParticles(self, prop_list)


class CommunicateSizes(Lowerable):
    def __init__(self, comm):
        super().__init__(comm.sim)
        self.comm = comm

    @pairs_inline
    def lower(self):
        Call_Void(self.sim, "pairs->communicateSizes", [self.comm.nsend, self.comm.nrecv])


class CommunicateData(Lowerable):
    def __init__(self, comm, prop_list):
        super().__init__(comm.sim)
        self.comm = comm
        self.prop_list = prop_list

    @pairs_inline
    def lower(self):
        elem_size = sum([self.sim.ndims() if p.type() == Types.Vector else 1 for p in self.prop_list])
        Call_Void(self.sim, "pairs->communicateData", [self.comm.send_buffer, self.comm.nsend,
                                                       self.comm.recv_buffer, self.comm.nrecv, elem_size])


class DetermineGhostParticles(Lowerable):
    def __init__(self, comm, step, spacing):
        super().__init__(comm.sim)
        self.comm = comm
        self.step = step
        self.spacing = spacing
        self.sim.add_statement(self)

    @pairs_device_block
    def lower(self):
        nsend_all = self.comm.nsend_all
        nsend = self.comm.nsend
        send_map = self.comm.send_map
        send_mult = self.comm.send_mult
        self.sim.module_name("determine_ghost_particles")
        self.sim.check_resize(self.comm.send_capacity, nsend)

        nb_rank_id = 0
        nsend_all.set(0)
        for i, _, pbc in self.comm.dom_part.ghost_particles(self.step, self.sim.position(), self.spacing):
            next_idx = AtomicAdd(self.sim, nsend_all, 1)
            send_map[next_idx].set(i)
            for d in range(self.sim.ndims()):
                send_mult[next_idx][d].set(pbc[d])

            nsend[nb_rank_id].add(1)
            nb_rank_id += 1


class PackGhostParticles(Lowerable):
    def __init__(self, comm, prop_list):
        super().__init__(comm.sim)
        self.comm = comm
        self.prop_list = prop_list
        self.sim.add_statement(self)

    def get_elems_per_particle(self):
        return sum([self.sim.ndims() if p.type() == Types.Vector else 1 for p in self.prop_list])

    @pairs_device_block
    def lower(self):
        send_buffer = self.comm.send_buffer
        send_map = self.comm.send_map
        send_mult = self.comm.send_mult
        elems_per_particle = self.get_elems_per_particle()
        self.sim.module_name("pack_ghost_particles" + "_".join([str(p.id()) for p in self.prop_list]))

        for i in For(self.sim, 0, self.comm.nsend):
            p_offset = 0
            m = send_map[i]
            buffer_index = i * elems_per_particle
            for p in self.prop_list:
                if p.type() == Types.Vector:
                    for d in range(self.sim.ndims()):
                        src = p[m][d]
                        if p == self.sim.position():
                            src += send_mult[i][d] * self.sim.grid.length(d)

                        send_buffer[buffer_index][p_offset + d].set(src)

                    p_offset += self.sim.ndims()

                else:
                    cast_fn = lambda x: Cast(self.sim, x, Types.Double) if p.type() != Types.Double else x
                    send_buffer[buffer_index][p_offset].set(cast_fn(p[m]))
                    p_offset += 1

            
class UnpackGhostParticles(Lowerable):
    def __init__(self, comm, prop_list):
        super().__init__(comm.sim)
        self.comm = comm
        self.prop_list = prop_list
        self.sim.add_statement(self)

    def get_elems_per_particle(self):
        return sum([self.sim.ndims() if p.type() == Types.Vector else 1 for p in self.prop_list])

    @pairs_device_block
    def lower(self):
        nlocal = self.sim.nlocal
        recv_buffer = self.comm.recv_buffer
        elems_per_particle = self.get_elems_per_particle()
        self.sim.module_name("unpack_ghost_particles" + "_".join([str(p.id()) for p in self.prop_list]))

        for i in For(self.sim, 0, self.comm.nrecv):
            p_offset = 0
            buffer_index = i * elems_per_particle
            for p in self.prop_list:
                if p.type() == Types.Vector:
                    for d in range(self.sim.ndims()):
                        p[nlocal + i][d].set(recv_buffer[buffer_index][p_offset + d])
                        
                    p_offset += self.sim.ndims()

                else:
                    cast_fn = lambda x: Cast(self.sim, x, p.type()) if p.type() != Types.Double else x
                    p[nlocal + i].set(cast_fn(recv_buffer[buffer_index][p_offset]))
                    p_offset += 1


class RemoveExchangedParticles_part1(Lowerable):
    def __init__(self, comm):
        super().__init__(comm.sim)
        self.comm = comm
        self.sim.add_statement(self)

    @pairs_host_block
    def lower(self):
        self.sim.module_name("remove_exchanged_particles_pt1")
        send_pos = self.sim.add_temp_var(self.sim.nparticles)
        for i in For(self.sim, 0, self.comm.nsend):
            for is_local in Branch(self.sim, self.comm.send_map[i] < self.sim.nlocal - self.comm.nsend):
                if is_local:
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

    @pairs_device_block
    def lower(self):
        self.sim.module_name("remove_exchanged_particles_pt2")
        for i in ParticleFor(self.sim):
            src = self.comm.exchg_copy_to[i]
            for _ in Filter(self.sim, src > 0):
                dst = self.comm.send_map[i]
                for p in self.prop_list:
                    if p.type() == Types.Vector:
                        for d in range(self.sim.ndims()):
                            p[dst][d].set(p[src][d])

                    else:
                        p[dst].set(p[src])

        self.sim.nlocal.set(self.sim.nlocal - self.comm.nsend)


class ChangeSizeAfterExchange(Lowerable):
    def __init__(self, comm):
        super().__init__(comm.sim)
        self.comm = comm
        self.sim.add_statement(self)

    @pairs_host_block
    def lower(self):
        sim = self.sim
        sim.module_name("change_size_after_exchange")
        sim.check_resize(self.sim.particle_capacity, self.sim.nlocal)
        self.sim.nlocal.set(sim.nlocal + self.comm.nrecv)
