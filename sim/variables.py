from ir.variables import VarDecl


class VariablesDecl:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        self.sim.clear_block()
        for v in self.sim.vars.all():
            VarDecl(self.sim, v)

        return self.sim.block
