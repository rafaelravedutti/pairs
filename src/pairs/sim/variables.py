from pairs.ir.block import pairs_block
from pairs.ir.variables import VarDecl
from pairs.sim.lowerable import Lowerable


class VariablesDecl(Lowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_block
    def lower(self):
        for v in self.sim.vars.all():
            VarDecl(self.sim, v)
