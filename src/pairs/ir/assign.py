from functools import reduce
from pairs.ir.ast_node import ASTNode
from pairs.ir.lit import Lit
from pairs.ir.types import Types
from pairs.ir.vector_expr import VectorExpression


class Assign(ASTNode):
    def __init__(self, sim, dest, src):
        super().__init__(sim)
        self.type = dest.type()
        src = Lit.cvt(sim, src)

        if dest.type() == Types.Vector:
            self.assignments = []

            for i in range(0, sim.ndims()):
                dim_src = src if not isinstance(src, VectorExpression) or src.type() != Types.Vector else src[i]
                self.assignments.append((dest[i], dim_src))
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
