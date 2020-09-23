from assign import AssignAST
from block import BlockAST
from loops import ForAST

class ParticleLattice():
    def __init__(self, sim, config, spacing, props, positions):
        self.sim = sim
        self.nparticles = 0
        self.config = config
        self.spacing = spacing
        self.props = props
        self.positions = positions

    def lower(self):
        assignments = []
        loops = []
        index = None
        nparticles = 1 

        for i in range(0, self.sim.dimensions):
            n = int((self.config[i][1] - self.config[i][0]) / self.spacing[i] - 0.001) + 1 
            loops.append(ForAST(self.sim, 0, n)) 
            if i > 0:
                loops[i - 1].set_body(BlockAST([loops[i]]))

            index = loops[i].iter() if index is None else index * n + loops[i].iter()
            nparticles *= n

        for i in range(0, self.sim.dimensions):
            pos = self.config[i][0] + self.spacing[i] * loops[i].iter()
            assignments.append(AssignAST(self, self.positions[index][i], pos))

        particle_props = self.sim.properties.defaults()
        for p in self.props:
            particle_props[p] = self.props[p]

        loops[self.sim.dimensions - 1].set_body(BlockAST(assignments))
        return (BlockAST(loops[0]), nparticles)
