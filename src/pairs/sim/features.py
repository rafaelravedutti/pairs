from pairs.ir.features import RegisterFeature
from pairs.sim.lowerable import FinalLowerable


class FeaturePropertiesAlloc(FinalLowerable):
    def __init__(self, sim):
        self.sim = sim

    @pairs_inline
    def lower(self):
        for fp in self.sim.feature_properties:
            RegisterFeatureProperty(self.sim, fp)
