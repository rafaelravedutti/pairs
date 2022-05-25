from pairs.ir.block import pairs_device_block, pairs_host_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import For, ParticleFor
from pairs.ir.utils import Print
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.lowerable import Lowerable


class PBC:
    def __init__(self, sim, grid, cutneigh, pbc_flags=[1, 1, 1]):
        self.sim = sim
        self.grid = grid
        self.cutneigh = cutneigh
        self.pbc_flags = pbc_flags
        self.npbc = sim.add_var('npbc', Types.Int32)
        self.pbc_capacity = sim.add_var('pbc_capacity', Types.Int32, 100)
        self.pbc_map = sim.add_array('pbc_map', [self.pbc_capacity], Types.Int32)
        self.pbc_mult = sim.add_array('pbc_mult', [self.pbc_capacity, sim.ndims()], Types.Int32)


class UpdatePBC(Lowerable):
    def __init__(self, sim, pbc):
        super().__init__(sim)
        self.pbc = pbc

    @pairs_device_block
    def lower(self):
        sim = self.sim
        ndims = sim.ndims()
        grid = self.pbc.grid
        npbc = self.pbc.npbc
        pbc_map = self.pbc.pbc_map
        pbc_mult = self.pbc.pbc_mult
        positions = self.pbc.sim.position()
        nlocal = self.pbc.sim.nlocal
        sim.module_name("update_pbc")

        for i in For(sim, 0, npbc):
            # TODO: allow syntax:
            # positions[nlocal + i].set(positions[pbc_map[i]] + pbc_mult[i] * grid.length)
            for d in range(0, ndims):
                positions[nlocal + i][d].set(positions[pbc_map[i]][d] + pbc_mult[i][d] * grid.length(d))


class EnforcePBC(Lowerable):
    def __init__(self, sim, pbc):
        super().__init__(sim)
        self.pbc = pbc

    @pairs_device_block
    def lower(self):
        sim = self.sim
        ndims = sim.ndims()
        grid = self.pbc.grid
        positions = sim.position()
        sim.module_name("enforce_pbc")

        for i in ParticleFor(sim):
            # TODO: VecFilter?
            for d in range(0, ndims):
                for _ in Filter(sim, positions[i][d] < grid.min(d)):
                    positions[i][d].add(grid.length(d))

                for _ in Filter(sim, positions[i][d] > grid.max(d)):
                    positions[i][d].sub(grid.length(d))


class SetupPBC(Lowerable):
    def __init__(self, sim, pbc):
        super().__init__(sim)
        self.pbc = pbc

    @pairs_host_block
    def lower(self):
        sim = self.sim
        ndims = sim.ndims()
        grid = self.pbc.grid
        cutneigh = self.pbc.cutneigh
        npbc = self.pbc.npbc
        pbc_capacity = self.pbc.pbc_capacity
        pbc_map = self.pbc.pbc_map
        pbc_mult = self.pbc.pbc_mult
        positions = self.pbc.sim.position()
        nlocal = self.pbc.sim.nlocal
        sim.module_name("setup_pbc")
        sim.check_resize(pbc_capacity, npbc)

        npbc.set(0)
        for d in range(0, ndims):
            for i in For(sim, 0, nlocal + npbc):
                pos = positions[i]
                grid_length = grid.length(d)

                # TODO: VecFilter?
                for _ in Filter(sim, pos[d] < grid.min(d) + cutneigh):
                    last_pos = positions[nlocal + npbc]
                    pbc_map[npbc].set(i)
                    pbc_mult[npbc][d].set(1)
                    last_pos[d].set(pos[d] + grid_length)

                    for d_ in [x for x in range(0, ndims) if x != d]:
                        pbc_mult[npbc][d_].set(0)
                        last_pos[d_].set(pos[d_])

                    npbc.add(1)

                for _ in Filter(sim, pos[d] > grid.max(d) - cutneigh):
                    last_pos = positions[nlocal + npbc]
                    pbc_map[npbc].set(i)
                    pbc_mult[npbc][d].set(-1)
                    last_pos[d].set(pos[d] - grid_length)

                    for d_ in [x for x in range(0, ndims) if x != d]:
                        pbc_mult[npbc][d_].set(0)
                        last_pos[d_].set(pos[d_])

                    npbc.add(1)
