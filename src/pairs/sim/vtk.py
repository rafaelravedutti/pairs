from pairs.ir.ast_node import ASTNode
from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Void
from pairs.ir.lit import Lit 
from pairs.sim.lowerable import Lowerable


class VTKWrite(Lowerable):
    def __init__(self, sim, filename, timestep, frequency):
        super().__init__(sim)
        self.filename = filename
        self.timestep = Lit.cvt(sim, timestep)
        self.frequency = frequency

    @pairs_inline
    def lower(self):
        nlocal = self.sim.nlocal
        nghost = self.sim.nghost
        nall = nlocal + nghost
        Call_Void(self.sim, "pairs::vtk_write_data", [self.filename + "_local", 0, nlocal, self.timestep, self.frequency])
        Call_Void(self.sim, "pairs::vtk_write_data", [self.filename + "_ghost", nlocal, nall, self.timestep, self.frequency])
