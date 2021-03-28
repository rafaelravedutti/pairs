from ir.ast_node import ASTNode
from ir.data_types import Type_Vector
from ir.lit import as_lit_ast
from functools import reduce


class Assign(ASTNode):
    def __init__(self, sim, dest, src):
        super().__init__(sim)
        self.type = dest.type()
        src = as_lit_ast(sim, src)

        if dest.type() == Type_Vector:
            self.assignments = []

            for i in range(0, sim.dimensions):
                from ir.bin_op import BinOp
                dim_src = src if not isinstance(src, BinOp) or src.type() != Type_Vector else src[i]
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
