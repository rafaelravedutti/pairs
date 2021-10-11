from pairs.ir.ast_node import ASTNode
from pairs.ir.functions import Call_Void
from pairs.ir.lit import as_lit_ast


class VTKWrite(ASTNode):
    def __init__(self, sim, filename, timestep):
        super().__init__(sim)
        self.filename = filename
        self.timestep = as_lit_ast(sim, timestep)

    def lower(self):
        nlocal = self.sim.nlocal
        npbc = self.sim.pbc.npbc
        self.sim.clear_block()
        nall = nlocal + npbc
        Call_Void(self.sim, "pairs::vtk_write_data", [self.filename + "_local", 0, nlocal, self.timestep])
        Call_Void(self.sim, "pairs::vtk_write_data", [self.filename + "_pbc", nlocal, nall, self.timestep])
        return self.sim.block

    def children(self):
        return [self.timestep]
