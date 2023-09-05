from pairs.ir.assign import Assign
from pairs.ir.block import pairs_device_block, pairs_host_block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import For, ParticleFor
from pairs.ir.utils import Print
from pairs.ir.select import Select
from pairs.ir.types import Types
from pairs.sim.lowerable import Lowerable


class EnforcePBC(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_device_block
    def lower(self):
        sim = self.sim
        grid = sim.grid
        ndims = sim.ndims()
        positions = sim.position()
        sim.module_name("enforce_pbc")

        for i in ParticleFor(sim):
            # TODO: VecFilter?
            for d in range(0, ndims):
                for _ in Filter(sim, positions[i][d] < grid.min(d)):
                    Assign(sim, positions[i][d], positions[i][d] + grid.length(d)) 

                for _ in Filter(sim, positions[i][d] > grid.max(d)):
                    Assign(sim, positions[i][d], positions[i][d] - grid.length(d)) 
