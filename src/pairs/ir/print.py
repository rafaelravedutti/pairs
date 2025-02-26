from pairs.ir.ast_node import ASTNode
from pairs.ir.lit import Lit

class Print(ASTNode):
    def __init__(self, sim, *args):
        super().__init__(sim)
        self.args = [Lit.cvt(sim, a) for a in args]
        self.sim.add_statement(self)

    def children(self):
        return self.args
    
    def __str__(self):
        return "Print<" + ", ".join(str(arg) for arg in self.args) + ">"

class PrintCode(ASTNode):
    def __init__(self, sim, str):
        super().__init__(sim)
        self.arg = Lit.cvt(sim, str)
        self.sim.add_statement(self)

    def children(self):
        return self.arg
    