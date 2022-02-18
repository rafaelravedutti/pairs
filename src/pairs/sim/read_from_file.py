from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Int
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
        self.sim.nlocal.set(Call_Int(self.sim, "pairs::read_particle_data",
                            [self.filename, self.grid_buffer, self.props, self.props.length()]))

        for d in range(self.sim.ndims()):
            self.grid.min(d).set(self.grid_buffer[d * 2 + 0])
            self.grid.max(d).set(self.grid_buffer[d * 2 + 1])
