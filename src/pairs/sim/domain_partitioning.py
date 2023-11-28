from pairs.ir.block import pairs_device_block, pairs_host_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import For, ParticleFor
from pairs.ir.utils import Print
from pairs.ir.scalars import ScalarOp
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.flags import Flags
from pairs.sim.lowerable import Lowerable


class DimensionRanges:
    def __init__(self, sim):
        self.sim            = sim
        self.neighbor_ranks = sim.add_static_array('neighbor_ranks', [sim.ndims() * 2], Types.Int32)
        self.pbc            = sim.add_static_array('pbc', [sim.ndims() * 2], Types.Int32)
        self.subdom         = sim.add_static_array('subdom', [sim.ndims() * 2], Types.Real)

    def min(self, dim):
        return self.subdom[dim * 2 + 0]

    def max(self, dim):
        return self.subdom[dim * 2 + 1]

    def number_of_steps(self):
        return self.sim.ndims()

    def step_indexes(self, step):
        return [step * 2 + 0, step * 2 + 1]

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
            j = step * 2 + 1
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
