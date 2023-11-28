from pairs.ir.assign import Assign
from pairs.ir.block import pairs_inline
from pairs.ir.loops import For
from pairs.ir.types import Types
from pairs.sim.lowerable import Lowerable


class ParticleLattice(Lowerable):
    def __init__(self, sim, grid, spacing, props, positions):
        super().__init__(sim)
        self.grid = grid
        self.spacing = spacing
        self.props = props
        self.positions = positions

    @pairs_inline
    def lower(self):
        index = None
        loop_indexes = []

        self.sim.clear_block()
        for _ in self.sim.nest_mode():
            for d in range(0, self.sim.ndims()):
                d_min, d_max = self.grid.range(d)
                n = int((d_max - d_min) / self.spacing[d] - 0.001) + 1

                for d_idx in For(self.sim, 0, n):
                    # index = (d_idx if index is None else index * n + d_idx)
                    loop_indexes.append(d_idx)

                    if d == self.sim.ndims() - 1:
                        index = self.sim.nlocal

                        for d_ in range(0, self.sim.ndims()):
                            pos = self.grid.min(d_) + self.spacing[d_] * loop_indexes[d_]
                            Assign(self.sim, self.positions[index][d_], pos)

                        for prop in [p for p in self.sim.properties.all()
                                     if p.volatile is False and p.name() != self.positions.name()]:
                            if prop.type() == Types.Vector:
                                for d_ in range(0, self.sim.ndims()):
                                    Assign(self.sim, prop[index][d_], prop.default()[d_])

                            else:
                                Assign(self.sim, prop[index], prop.default())

                        Assign(self.sim, self.sim.nlocal, self.sim.nlocal + 1)

        return self.sim.block
