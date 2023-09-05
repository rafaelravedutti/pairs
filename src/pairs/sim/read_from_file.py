from pairs.ir.assign import Assign
from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Int, Call_Void
from pairs.ir.particle_attributes import ParticleAttributeList
from pairs.ir.types import Types
from pairs.sim.grid import MutableGrid
from pairs.sim.lowerable import Lowerable


class ReadParticleData(Lowerable):
    def __init__(self, sim, filename, items):
        super().__init__(sim)
        self.filename = filename
        self.attrs = ParticleAttributeList(sim, items)
        self.grid = MutableGrid(sim, sim.ndims())
        self.grid_buffer = self.sim.add_static_array("grid_buffer", [self.sim.ndims() * 2], Types.Double)

    @pairs_inline
    def lower(self):
        Call_Void(self.sim, "pairs::read_grid_data", [self.filename, self.grid_buffer])
        for d in range(self.sim.ndims()):
            Assign(self.sim, self.grid.min(d), self.grid_buffer[d * 2 + 0])
            Assign(self.sim, self.grid.max(d), self.grid_buffer[d * 2 + 1])

        dom_part = self.sim.domain_partitioning()
        grid_array = [[self.grid.min(d), self.grid.max(d)] for d in range(self.sim.ndims())]
        Call_Void(self.sim, "pairs->initDomain", [param for delim in grid_array for param in delim]),
        Call_Void(self.sim, "pairs->fillCommunicationArrays", [dom_part.neighbor_ranks, dom_part.pbc, dom_part.subdom])
        Assign(self.sim, self.sim.nlocal, Call_Int(self.sim, "pairs::read_particle_data", [self.filename, self.attrs, self.attrs.length()]))


class ReadFeatureData(Lowerable):
    def __init__(self, sim, filename, feature, items):
        super().__init__(sim)
        self.filename = filename
        self.feature = feature
        self.attrs = ParticleAttributeList(sim, items)

    @pairs_inline
    def lower(self):
        Call_Int(self.sim, "pairs::read_feature_data", [self.filename, self.feature.id(), self.attrs, self.attrs.length()])
