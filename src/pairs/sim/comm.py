from pairs.ir.block import pairs_device_block, pairs_host_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import For, ParticleFor
from pairs.ir.utils import Print
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.lowerable import Lowerable


class Comm:
    def __init__(self, sim):
        self.sim = sim
        self.nghost             =   sim.add_var('nghost', Types.Int32)
        self.ghost_capacity     =   sim.add_var('ghost_capacity', Types.Int32, 100)
        self.ghost_map          =   sim.add_array('ghost_map', [self.ghost_capacity], Types.Int32)
        self.ghost_mult         =   sim.add_array('ghost_mult', [self.ghost_capacity, sim.ndims()], Types.Int32)


class DetermineGhostParticles(Lowerable):
    def __init__(self, sim, comm):
        super().__init__(sim)
        self.comm = comm

    @pairs_host_block
    def lower(self):
        sim = self.sim
        ndims = sim.ndims()
        grid = self.sim.grid
        cutneigh = self.sim.cell_spacing()
        nghost = self.comm.nghost
        ghost_capacity = self.comm.ghost_capacity
        ghost_map = self.comm.ghost_map
        ghost_mult = self.comm.ghost_mult
        positions = self.sim.position()
        nlocal = self.sim.nlocal
        sim.module_name("setup_comm")
        sim.check_resize(ghost_capacity, nghost)

        nghost.set(0)
        for d in range(0, ndims):
            for i in For(sim, 0, nlocal + nghost):
                pos = positions[i]
                grid_length = grid.length(d)

                # TODO: VecFilter?
                for _ in Filter(sim, pos[d] < grid.min(d) + cutneigh):
                    last_pos = positions[nlocal + nghost]
                    ghost_map[nghost].set(i)
                    ghost_mult[nghost][d].set(1)
                    last_pos[d].set(pos[d] + grid_length)

                    for d_ in [x for x in range(0, ndims) if x != d]:
                        ghost_mult[nghost][d_].set(0)
                        last_pos[d_].set(pos[d_])

                    nghost.add(1)

                for _ in Filter(sim, pos[d] > grid.max(d) - cutneigh):
                    last_pos = positions[nlocal + nghost]
                    ghost_map[nghost].set(i)
                    ghost_mult[nghost][d].set(-1)
                    last_pos[d].set(pos[d] - grid_length)

                    for d_ in [x for x in range(0, ndims) if x != d]:
                        ghost_mult[nghost][d_].set(0)
                        last_pos[d_].set(pos[d_])

                    nghost.add(1)


class UpdateGhostParticles(Lowerable):
    def __init__(self, sim, comm):
        super().__init__(sim)
        self.comm = comm

    @pairs_device_block
    def lower(self):
        sim = self.sim
        ndims = sim.ndims()
        grid = self.sim.grid
        nghost = self.comm.nghost
        ghost_map = self.comm.ghost_map
        ghost_mult = self.comm.ghost_mult
        positions = self.sim.position()
        nlocal = self.sim.nlocal
        sim.module_name("update_comm")

        for i in For(sim, 0, nghost):
            # TODO: allow syntax:
            # positions[nlocal + i].set(positions[ghost_map[i]] + ghost_mult[i] * grid.length)
            for d in range(0, ndims):
                positions[nlocal + i][d].set(positions[ghost_map[i]][d] + ghost_mult[i][d] * grid.length(d))


class ExchangeParticles(Lowerable):
    def __init__(self, sim, comm):
        super().__init__(sim)
        self.comm = comm
