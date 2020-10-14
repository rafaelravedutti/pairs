from ast.sizeof import Sizeof
from functools import reduce
import operator


class Malloc:
    def __init__(self, sim, array, a_type, sizes, decl=False):
        self.sim = sim
        self.parent_block = None
        self.array = array
        self.array_type = a_type
        self.decl = decl
        self.prim_size = Sizeof(sim, a_type)
        self.size = self.prim_size * (
            reduce(operator.mul, sizes) if isinstance(sizes, list) else sizes)
        self.sim.add_statement(self)

    def children(self):
        return [self.array, self.size]

    def generate(self, mem=False):
        self.sim.code_gen.generate_malloc(
            self.array.generate(),
            self.array_type,
            self.size.generate_inline(recursive=True),
            self.decl)

    def transform(self, fn):
        self.array = self.array.transform(fn)
        self.size = self.size.transform(fn)
        return fn(self)


class Realloc:
    def __init__(self, sim, array, a_type, size):
        self.sim = sim
        self.parent_block = None
        self.array = array
        self.array_type = a_type
        self.size = size
        self.sim.add_statement(self)

    def children(self):
        return [self.array, self.size]

    def generate(self, mem=False):
        self.sim.code_gen.generate_realloc(
            self.array.generate(), self.array_type, self.size.generate())

    def transform(self, fn):
        self.array = self.array.transform(fn)
        self.size = self.size.transform(fn)
        return fn(self)
