from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Int, Call_Void
from pairs.ir.properties import PropertyList
from pairs.ir.types import Types
from pairs.sim.grid import MutableGrid
from pairs.sim.lowerable import Lowerable


class ReadFromFile(Lowerable):
    def __init__(self, sim, filename, props):
        super().__init__(sim)
        self.filename = filename
        self.props = PropertyList(sim, props)
        self.grid = MutableGrid(sim, sim.ndims())
        self.grid_buffer = self.sim.add_static_array("grid_buffer", [self.sim.ndims() * 2], Types.Double)

    @pairs_inline
    def lower(self):
        Call_Void(self.sim, "pairs::read_grid_data", [self.filename, self.grid_buffer])
        for d in range(self.sim.ndims()):
            self.grid.min(d).set(self.grid_buffer[d * 2 + 0])
            self.grid.max(d).set(self.grid_buffer[d * 2 + 1])

        dom_part = self.sim.domain_partitioning()
        grid_array = [[self.grid.min(d), self.grid.max(d)] for d in range(self.sim.ndims())]
        Call_Void(self.sim, "pairs->initDomain", [param for delim in grid_array for param in delim]),
        Call_Void(self.sim, "pairs->fillCommunicationArrays", [dom_part.neighbor_ranks, dom_part.pbc, dom_part.subdom])
        self.sim.nlocal.set(Call_Int(self.sim, "pairs::read_particle_data", [self.filename, self.props, self.props.length()]))
