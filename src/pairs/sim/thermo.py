from pairs.ir.assign import Assign
from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Int, Call_Void
from pairs.ir.particle_attributes import ParticleAttributeList
from pairs.ir.types import Types
from pairs.sim.grid import Grid3D
from pairs.sim.lowerable import Lowerable


class ComputeThermo(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        xprd = self.sim.grid.length(0)
        yprd = self.sim.grid.length(1)
        zprd = self.sim.grid.length(2)
        Call_Void(self.sim, "pairs::compute_thermo", [self.sim.nlocal, xprd, yprd, zprd, 1])
