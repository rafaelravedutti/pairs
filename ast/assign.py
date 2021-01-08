from ast.ast_node import ASTNode
from ast.data_types import Type_Vector
from ast.lit import as_lit_ast
from functools import reduce


class Assign(ASTNode):
    def __init__(self, sim, dest, src):
        super().__init__(sim)
        self.parent_block = None
        self.type = dest.type()
        src = as_lit_ast(sim, src)

        if dest.type() == Type_Vector:
            self.assignments = []

            for i in range(0, sim.dimensions):
                from ast.bin_op import BinOp
                dsrc = (src if (not isinstance(src, BinOp) or
                                src.type() != Type_Vector)
                        else src[i])

                self.assignments.append((dest[i], dsrc))
        else:
            self.assignments = [(dest, src)]

    def __str__(self):
        return f"Assign<{self.assignments}>"

    def children(self):
        return reduce((lambda x, y: x + y), [
                      [self.assignments[i][0], self.assignments[i][1]]
                      for i in range(0, len(self.assignments))])

    def transform(self, fn):
        self.assignments = [(
            self.assignments[i][0].transform(fn),
            self.assignments[i][1].transform(fn))
            for i in range(0, len(self.assignments))]

        return fn(self)
