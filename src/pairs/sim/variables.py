from pairs.ir.block import pairs_block
from pairs.ir.variables import VarDecl


class VariablesDecl:
    def __init__(self, sim):
        self.sim = sim

    @pairs_block
    def lower(self):
        for v in self.sim.vars.all():
            VarDecl(self.sim, v)
