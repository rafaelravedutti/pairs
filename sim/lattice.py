from ast.assign import AssignAST
from ast.block import BlockAST
from ast.loops import ForAST


class ParticleLattice():
    def __init__(self, sim, config, spacing, props, positions):
        self.sim = sim
        self.nparticles = 0
        self.config = config
        self.spacing = spacing
        self.props = props
        self.positions = positions

    def lower(self):
        dims = self.sim.dimensions
        assignments = []
        loops = []
        index = None
        nparticles = 1

        for i in range(0, dims):
            dim_cfg = self.config[i]
            n = int((dim_cfg[1] - dim_cfg[0]) / self.spacing[i] - 0.001) + 1
            loops.append(ForAST(self.sim, 0, n))
            if i > 0:
                loops[i - 1].set_body(BlockAST(self.sim, [loops[i]]))

            index = (loops[i].iter() if index is None
                     else index * n + loops[i].iter())
            nparticles *= n

        for i in range(0, dims):
            pos = self.config[i][0] + self.spacing[i] * loops[i].iter()
            assignments.append(
                AssignAST(self.sim, self.positions[index][i], pos))

        particle_props = self.sim.properties.defaults()
        for p in self.props:
            particle_props[p] = self.props[p]

        loops[dims - 1].set_body(BlockAST(self.sim, assignments))
        return (BlockAST(self.sim, loops[0]), nparticles)
