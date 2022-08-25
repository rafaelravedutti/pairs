from pairs.ir.atomic import AtomicAdd
from pairs.ir.block import pairs_device_block, pairs_host_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.cast import Cast
from pairs.ir.loops import For, ParticleFor
from pairs.ir.utils import Print
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.lowerable import Lowerable


class Comm:
    def __init__(self, sim, max_neigh_ranks=6, max_buffer_elems=7):
        self.sim = sim
        self.nghost        = sim.add_var('nghost', Types.Int32)
        self.nsend_all     = sim.add_var('nsend_all', Types.Int32)
        self.send_capacity = sim.add_var('send_capacity', Types.Int32, 100)
        self.recv_capacity = sim.add_var('recv_capacity', Types.Int32, 100)
        self.nsend         = sim.add_array('nsend', [max_neigh_ranks], Types.Int32)
        self.send_buffer   = sim.add_array('send_buffer', [self.send_capacity, max_buffer_elems], Types.Double)
        self.send_map      = sim.add_array('send_map', [self.send_capacity], Types.Int32)
        self.send_mult     = sim.add_array('send_mult', [self.send_capacity, sim.ndims()], Types.Int32)
        self.nrecv         = sim.add_array('nrecv', [max_neigh_ranks], Types.Int32)
        self.recv_buffer   = sim.add_array('recv_buffer', [self.recv_capacity, max_buffer_elems], Types.Double)
        self.recv_map      = sim.add_array('recv_map', [self.recv_capacity], Types.Int32)
        self.recv_mult     = sim.add_array('recv_mult', [self.recv_capacity, sim.ndims()], Types.Int32)


class DetermineGhostParticles(Lowerable):
    def __init__(self, sim, comm, dom_part):
        super().__init__(sim)
        self.comm = comm
        self.dom_part = dom_part

    @pairs_device_block
    def lower(self):
        nsend = self.comm.nsend
        send_map = self.comm.send_map
        send_mult = self.comm.send_mult
        sim.module_name("determine_ghost_particles")
        sim.check_resize(self.comm.send_capacity, nsend)

        nb_rank_id = 0
        nsend_all.set(0)
        for i, _, pbc in self.dom_part.ghost_particles(self.sim.position(), self.sim.cell_spacing()):
            n = AtomicAdd(self.sim, nsend_all, 1)
            send_map[n].set(i)
            for d in self.sim.ndims():
                send_mult[n][d].set(pbc[d])

            self.nsend[nb_rank_id].add(1)
            nb_rank_id += 1


class PackGhostParticles(Lowerable):
    def __init__(self, sim, comm, prop_list):
        super().__init__(sim)
        self.comm = comm
        self.prop_list = prop_list

    def get_elems_per_particle(self):
        return sum([self.sim.ndims() if p.type() == Types.Vector else 1 for p in self.prop_list])

    @pairs_device_block
    def lower(self):
        send_buffer = self.comm.send_buffer
        send_map = self.comm.send_map
        elems_per_particle = self.get_elems_per_particle()
        sim.module_name("pack_ghost_particles" + sum(["_{p.id()}" for p in self.prop_list]))

        for i in For(self.sim, self.comm.nsend):
            p_offset = 0
            for p in self.prop_list:
                if p.type() == Types.Vector:
                    for d in self.sim.ndims():
                        src = p[send_map[i]][d]
                        if p == self.sim.position():
                            src += send_mult[i][d] * grid.length(d)

                        send_buffer[i * elems_per_particle + p_offset + d].set(src)

                    p_offset += self.sim.ndims()

                else:
                    cast_fn = lambda x: Cast(self.sim, x, Types.Double) if p.type() != Types.Double else x
                    send_buffer[i * elems_per_particle + p_offset].set(cast_fn(p[send_map[i]]))
                    p_offset += 1

            
class UnpackGhostParticles(Lowerable):
    def __init__(self, sim, comm, prop_list):
        super().__init__(sim)
        self.comm = comm
        self.prop_list = prop_list

    def get_elems_per_particle(self):
        return sum([self.sim.ndims() if p.type() == Types.Vector else 1 for p in self.prop_list])

    @pairs_device_block
    def lower(self):
        nlocal = self.sim.nlocal
        recv_buffer = self.comm.recv_buffer
        elems_per_particle = self.get_elems_per_particle()
        sim.module_name("unpack_ghost_particles" + sum(["_{p.id()}" for p in self.prop_list]))

        for i in For(self.sim, self.comm.nrecv):
            p_offset = 0
            for p in self.prop_list:
                if p.type() == Types.Vector:
                    for d in self.sim.ndims():
                        p[nlocal + i][d].set(recv_buffer[i * elems_per_particle + p_offset + d])
                        
                    p_offset += self.sim.ndims()

                else:
                    cast_fn = lambda x: Cast(self.sim, x, p.type()) if p.type() != Types.Double else x
                    p[nlocal + i].set(cast_fn(recv_buffer[i * elems_per_particle + p_offset]))
                    p_offset += 1


class ExchangeParticles(Lowerable):
    def __init__(self, sim, comm):
        super().__init__(sim)
        self.comm = comm
