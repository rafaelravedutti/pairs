from ast.branches import Branch, Filter
from ast.data_types import Type_Int
from ast.loops import For, ParticleFor


class PBC:
    def __init__(self, sim, grid, cutneigh, pbc_flags=[1, 1, 1]):
        self.sim = sim
        self.grid = grid
        self.cutneigh = cutneigh
        self.pbc_flags = pbc_flags
        self.npbc = sim.add_var('npbc', Type_Int)
        self.pbc_capacity = sim.add_var('pbc_capacity', Type_Int, 20)
        self.pbc_map = sim.add_array('pbc_map', [self.pbc_capacity], Type_Int)
        self.pbc_mult = sim.add_array(
            'pbc_mult', [self.pbc_capacity, sim.dimensions], Type_Int)


class UpdatePBC:
    def __init__(self, pbc):
        self.pbc = pbc

    def lower(self):
        pbc = self.pbc
        positions = pbc.sim.property('position')
        nlocal = pbc.sim.nparticles

        pbc.sim.clear_block()
        for i in For(pbc.sim, 0, pbc.npbc):
            for d in range(0, pbc.sim.dimensions):
                positions[nlocal + i][d].set(
                    positions[pbc.pbc_map[i]][d] +
                    pbc.pbc_mult[i][d] * pbc.grid.length(d))

        return pbc.sim.block


class EnforcePBC:
    def __init__(self, pbc):
        self.pbc = pbc

    def lower(self):
        pbc = self.pbc
        positions = pbc.sim.property('position')

        pbc.sim.clear_block()
        for i in ParticleFor(pbc.sim):
            for d in range(0, pbc.sim.dimensions):
                for _ in Filter(pbc.sim, positions[i][d] < pbc.grid.min(d)):
                    positions[i][d].add(pbc.grid.length(d))

                for _ in Filter(pbc.sim, positions[i][d] > pbc.grid.max(d)):
                    positions[i][d].sub(pbc.grid.length(d))

        return pbc.sim.block


class SetupPBC:
    def __init__(self, pbc):
        self.pbc = pbc

    def lower(self):
        pbc = self.pbc
        positions = pbc.sim.property('position')
        nlocal = pbc.sim.nparticles

        pbc.sim.clear_block()
        pbc.npbc.set(0)
        for d in range(0, pbc.sim.dimensions):
            for i in For(pbc.sim, 0, nlocal + pbc.npbc):
                lower_cond = positions[i][d] < pbc.grid.min(d) + pbc.cutneigh
                upper_cond = positions[i][d] > pbc.grid.max(d) - pbc.cutneigh

                for _ in Filter(pbc.sim, lower_cond):
                    pbc.pbc_map[pbc.npbc].set(i)
                    pbc.pbc_mult[pbc.npbc][d].set(1)

                    for cond in Branch(pbc.sim, i < nlocal):
                        if cond:
                            for d_ in range(0, pbc.sim.dimensions):
                                if d_ != d:
                                    pbc.pbc_mult[pbc.npbc][d_].set(0)
                        else:
                            for d_ in range(0, pbc.sim.dimensions):
                                if d_ != d:
                                    pbc.pbc_mult[pbc.npbc][d_].set(
                                        pbc.pbc_mult[i - nlocal][d_])

                    pbc.npbc.add(1)

                for _ in Filter(pbc.sim, upper_cond):
                    pbc.pbc_map[pbc.npbc].set(i)
                    pbc.pbc_mult[pbc.npbc][d].set(-1)

                    for cond in Branch(pbc.sim, i < nlocal):
                        if cond:
                            for d_ in range(0, pbc.sim.dimensions):
                                if d_ != d:
                                    pbc.pbc_mult[pbc.npbc][d_].set(0)
                        else:
                            for d_ in range(0, pbc.sim.dimensions):
                                if d_ != d:
                                    pbc.pbc_mult[pbc.npbc][d_].set(
                                        pbc.pbc_mult[i - nlocal][d_])

                    pbc.npbc.add(1)

        return pbc.sim.block
