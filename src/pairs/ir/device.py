from functools import reduce
import operator
from pairs.ir.ast_node import ASTNode
from pairs.ir.bin_op import BinOp
from pairs.ir.sizeof import Sizeof


class CopyToDevice(ASTNode):
    def __init__(self, sim, prop):
        super().__init__(sim)
        sizes = prop.sizes()
        self.prop = prop
        self.prim_size = Sizeof(sim, prop.type())
        self.size = BinOp.inline(self.prim_size * (reduce(operator.mul, sizes) if isinstance(sizes, list) else sizes))

    def children(self):
        return [self.prop]


class CopyToHost(ASTNode):
    def __init__(self, sim, prop):
        super().__init__(sim)
        sizes = prop.sizes()
        self.prop = prop
        self.prim_size = Sizeof(sim, prop.type())
        self.size = BinOp.inline(self.prim_size * (reduce(operator.mul, sizes) if isinstance(sizes, list) else sizes))

    def children(self):
        return [self.prop]
