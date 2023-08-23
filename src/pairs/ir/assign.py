from functools import reduce
from pairs.ir.ast_node import ASTNode
from pairs.ir.lit import Lit
from pairs.ir.types import Types


class Assign(ASTNode):
    def __init__(self, sim, dest, src):
        super().__init__(sim)
        src = Lit.cvt(sim, src)

        if dest.type() == Types.Vector:
            self.assignments = \
                [(dest[d], src if src.type() != Types.Vector else src[d]) for d in range(sim.ndims())]

        else:
            self.assignments = [(dest, src)]

    def __str__(self):
        return f"Assign<{self.assignments}>"

    def destinations(self):
        return [a[0] for a in self.assignments]

    def sources(self):
        return [a[1] for a in self.assignments]

    def children(self):
        return reduce((lambda x, y: x + y), [[a[0], a[1]] for a in self.assignments])
