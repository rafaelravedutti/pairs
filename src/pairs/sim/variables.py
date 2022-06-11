from pairs.ir.block import pairs_inline
from pairs.ir.variables import VarDecl
from pairs.sim.lowerable import FinalLowerable


class VariablesDecl(FinalLowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        for v in self.sim.vars.all():
            VarDecl(self.sim, v)
