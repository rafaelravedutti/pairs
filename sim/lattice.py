from ast.loops import ForAST


class ParticleLattice():
    def __init__(self, sim, config, spacing, props, positions):
        self.sim = sim
        self.config = config
        self.spacing = spacing
        self.props = props
        self.positions = positions

    def lower(self):
        index = None
        loop_indexes = []

        self.sim.clear_block()
        for _ in self.sim.nest_mode():
            for d in range(0, self.sim.dimensions):
                d_cfg = self.config[d]
                n = int((d_cfg[1] - d_cfg[0]) / self.spacing[d] - 0.001) + 1

                for d_idx in ForAST(self.sim, 0, n):
                    index = (d_idx if index is None else index * n + d_idx)
                    loop_indexes.append(d_idx)

                    if d == self.sim.dimensions - 1:
                        for d_ in range(0, self.sim.dimensions):
                            pos = self.config[d_][0] + \
                                  self.spacing[d_] * loop_indexes[d_]
                            self.positions[index][d_].set(pos)

                        self.sim.nparticles.set(self.sim.nparticles + 1)

        return self.sim.block
