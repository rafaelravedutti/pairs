from pairs.ir.block import pairs_block
from pairs.ir.data_types import Type_Vector
from pairs.ir.loops import For


class ParticleLattice():
    def __init__(self, sim, grid, spacing, props, positions):
        self.sim = sim
        self.grid = grid
        self.spacing = spacing
        self.props = props
        self.positions = positions

    @pairs_block
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
                            self.positions[index][d_].set(pos)

                        for prop in [p for p in self.sim.properties.all()
                                     if p.volatile is False and p.name() != self.positions.name()]:
                            if prop.type() == Type_Vector:
                                for d_ in range(0, self.sim.ndims()):
                                    prop[index][d_].set(prop.default()[d_])

                            else:
                                prop[index].set(prop.default())

                        self.sim.nlocal.set(self.sim.nlocal + 1)

        return self.sim.block
