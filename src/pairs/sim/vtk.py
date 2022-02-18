from pairs.ir.ast_node import ASTNode
from pairs.ir.block import pairs_inline
from pairs.ir.functions import Call_Void
from pairs.ir.lit import Lit 
from pairs.sim.lowerable import Lowerable


class VTKWrite(Lowerable):
    def __init__(self, sim, filename, timestep):
        super().__init__(sim)
        self.filename = filename
        self.timestep = Lit.cvt(sim, timestep)

    @pairs_inline
    def lower(self):
        nlocal = self.sim.nlocal
        npbc = self.sim.pbc.npbc
        nall = nlocal + npbc
        Call_Void(self.sim, "pairs::vtk_write_data", [self.filename + "_local", 0, nlocal, self.timestep])
        Call_Void(self.sim, "pairs::vtk_write_data", [self.filename + "_pbc", nlocal, nall, self.timestep])
