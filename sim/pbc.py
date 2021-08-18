from ir.branches import Branch, Filter
from ir.data_types import Type_Int
from ir.loops import For, ParticleFor
from ir.utils import Print
from ir.select import Select
from sim.resize import Resize


class PBC:
    def __init__(self, sim, grid, cutneigh, pbc_flags=[1, 1, 1]):
        self.sim = sim
        self.grid = grid
        self.cutneigh = cutneigh
        self.pbc_flags = pbc_flags
        self.npbc = sim.add_var('npbc', Type_Int)
        self.pbc_capacity = sim.add_var('pbc_capacity', Type_Int, 100)
        self.pbc_map = sim.add_array('pbc_map', [self.pbc_capacity], Type_Int)
        self.pbc_mult = sim.add_array('pbc_mult', [self.pbc_capacity, sim.ndims()], Type_Int)


class UpdatePBC:
    def __init__(self, pbc):
        self.pbc = pbc

    def lower(self):
        sim = self.pbc.sim
        ndims = sim.ndims()
        grid = self.pbc.grid
        npbc = self.pbc.npbc
        pbc_map = self.pbc.pbc_map
        pbc_mult = self.pbc.pbc_mult
        positions = self.pbc.sim.property('position')
        nlocal = self.pbc.sim.nlocal

        sim.clear_block()
        sim.add_statement(Print(sim, "UpdatePBC"))
        for i in For(sim, 0, npbc):
            # TODO: allow syntax:
            # positions[nlocal + i].set(positions[pbc_map[i]] + pbc_mult[i] * grid.length)
            for d in range(0, ndims):
                positions[nlocal + i][d].set(positions[pbc_map[i]][d] + pbc_mult[i][d] * grid.length(d))

        return sim.block


class EnforcePBC:
    def __init__(self, pbc):
        self.pbc = pbc

    def lower(self):
        sim = self.pbc.sim
        ndims = sim.ndims()
        grid = self.pbc.grid
        positions = sim.property('position')

        sim.clear_block()
        sim.add_statement(Print(sim, "EnforcePBC"))
        for i in ParticleFor(sim):
            # TODO: VecFilter?
            for d in range(0, ndims):
                for _ in Filter(sim, positions[i][d] < grid.min(d)):
                    positions[i][d].add(grid.length(d))

                for _ in Filter(sim, positions[i][d] > grid.max(d)):
                    positions[i][d].sub(grid.length(d))

        return sim.block


class SetupPBC:
    def __init__(self, pbc):
        self.pbc = pbc

    def lower(self):
        sim = self.pbc.sim
        ndims = sim.ndims()
        grid = self.pbc.grid
        cutneigh = self.pbc.cutneigh
        npbc = self.pbc.npbc
        pbc_capacity = self.pbc.pbc_capacity
        pbc_map = self.pbc.pbc_map
        pbc_mult = self.pbc.pbc_mult
        positions = self.pbc.sim.property('position')
        nlocal = self.pbc.sim.nlocal

        sim.clear_block()
        sim.add_statement(Print(sim, "SetupPBC"))
        for resize in Resize(sim, pbc_capacity):
            npbc.set(0)
            for d in range(0, ndims):
                for i in For(sim, 0, nlocal + npbc):
                    #last_id = nlocal + npbc
                    # TODO: VecFilter?
                    for _ in Filter(sim, positions[i][d] < grid.min(d) + cutneigh):
                        for capacity_exceeded in Branch(sim, npbc >= pbc_capacity):
                            if capacity_exceeded:
                                resize.set(Select(sim, resize > npbc, resize + 1, npbc))
                            else:
                                last_id = nlocal + npbc
                                pbc_map[npbc].set(i)
                                pbc_mult[npbc][d].set(1)
                                positions[last_id][d].set(positions[i][d] + grid.length(d))

                                for d_ in [x for x in range(0, ndims) if x != d]:
                                    pbc_mult[npbc][d_].set(0)
                                    positions[last_id][d_].set(positions[i][d_])

                                npbc.add(1)

                    for _ in Filter(sim, positions[i][d] > grid.max(d) - cutneigh):
                        for capacity_exceeded in Branch(sim, npbc >= pbc_capacity):
                            if capacity_exceeded:
                                resize.set(Select(sim, resize > npbc, resize + 1, npbc))
                            else:
                                last_id = nlocal + npbc
                                pbc_map[npbc].set(i)
                                pbc_mult[npbc][d].set(-1)
                                positions[last_id][d].set(positions[i][d] - grid.length(d))

                                for d_ in [x for x in range(0, ndims) if x != d]:
                                    pbc_mult[npbc][d_].set(0)
                                    positions[last_id][d_].set(positions[i][d_])

                                npbc.add(1)

        return sim.block
