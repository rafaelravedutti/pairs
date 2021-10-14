from pairs.ir.ast_node import ASTNode
from pairs.ir.block import pairs_block
from pairs.ir.functions import Call_Void
from pairs.ir.lit import as_lit_ast
from pairs.sim.lowerable import Lowerable


class VTKWrite(Lowerable):
    def __init__(self, sim, filename, timestep):
        super().__init__(sim)
        self.filename = filename
        self.timestep = as_lit_ast(sim, timestep)

    @pairs_block
    def lower(self):
        nlocal = self.sim.nlocal
        npbc = self.sim.pbc.npbc
        nall = nlocal + npbc
        Call_Void(self.sim, "pairs::vtk_write_data", [self.filename + "_local", 0, nlocal, self.timestep])
        Call_Void(self.sim, "pairs::vtk_write_data", [self.filename + "_pbc", nlocal, nall, self.timestep])
