from ir.ast_node import ASTNode
from ir.bin_op import BinOp
from ir.sizeof import Sizeof
from functools import reduce
import operator


class Malloc(ASTNode):
    def __init__(self, sim, array, sizes, decl=False):
        super().__init__(sim)
        self.array = array
        self.decl = decl
        self.prim_size = Sizeof(sim, array.type())
        self.size = BinOp.inline(self.prim_size * (reduce(operator.mul, sizes) if isinstance(sizes, list) else sizes))
        self.sim.add_statement(self)

    def children(self):
        return [self.array, self.size]


class Realloc(ASTNode):
    def __init__(self, sim, array, size):
        super().__init__(sim)
        self.array = array
        self.prim_size = Sizeof(sim, array.type())
        self.size = BinOp.inline(self.prim_size * size)
        self.sim.add_statement(self)

    def children(self):
        return [self.array, self.size]
