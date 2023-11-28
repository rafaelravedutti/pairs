from pairs.ir.block import pairs_device_block, pairs_inline
from pairs.ir.features import RegisterFeatureProperty
from pairs.sim.lowerable import FinalLowerable


class AllocateFeatureProperties(FinalLowerable):
    def __init__(self, sim):
        self.sim = sim

    @pairs_inline
    def lower(self):
        for fp in self.sim.feature_properties:
            RegisterFeatureProperty(self.sim, fp)
