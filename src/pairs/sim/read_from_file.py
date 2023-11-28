from pairs.ir.assign import Assign
from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Int, Call_Void
from pairs.ir.particle_attributes import ParticleAttributeList
from pairs.ir.types import Types
from pairs.sim.grid import MutableGrid
from pairs.sim.lowerable import Lowerable


class ReadParticleData(Lowerable):
    def __init__(self, sim, filename, items, shape_id):
        super().__init__(sim)
        self.filename = filename
        self.attrs = ParticleAttributeList(sim, items)
        self.shape_id = shape_id

    @pairs_inline
    def lower(self):
        Assign(self.sim, self.sim.nlocal, Call_Int(self.sim, "pairs::read_particle_data", [self.filename, self.attrs, self.attrs.length(), self.shape_id, self.sim.nlocal]))


class ReadFeatureData(Lowerable):
    def __init__(self, sim, filename, feature, items):
        super().__init__(sim)
        self.filename = filename
        self.feature = feature
        self.attrs = ParticleAttributeList(sim, items)

    @pairs_inline
    def lower(self):
        Call_Int(self.sim, "pairs::read_feature_data", [self.filename, self.feature.id(), self.attrs, self.attrs.length()])
