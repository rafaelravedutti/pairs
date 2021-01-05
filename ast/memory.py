from ast.expr import BinOp
from ast.sizeof import Sizeof
from functools import reduce
import operator


class Malloc:
    def __init__(self, sim, array, sizes, decl=False):
        self.sim = sim
        self.parent_block = None
        self.array = array
        self.decl = decl
        self.prim_size = Sizeof(sim, array.type())
        self.size = BinOp.inline(self.prim_size * (reduce(operator.mul, sizes) if isinstance(sizes, list) else sizes))
        self.sim.add_statement(self)

    def children(self):
        return [self.array, self.size]

    def transform(self, fn):
        self.array = self.array.transform(fn)
        self.size = self.size.transform(fn)
        return fn(self)


class Realloc:
    def __init__(self, sim, array, size):
        self.sim = sim
        self.parent_block = None
        self.array = array
        self.prim_size = Sizeof(sim, array.type())
        self.size = BinOp.inline(self.prim_size * size)
        self.sim.add_statement(self)

    def children(self):
        return [self.array, self.size]

    def transform(self, fn):
        self.array = self.array.transform(fn)
        self.size = self.size.transform(fn)
        return fn(self)
