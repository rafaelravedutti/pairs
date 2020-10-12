from ast.sizeof import SizeofAST
from functools import reduce
import operator


class MallocAST:
    def __init__(self, sim, array, a_type, sizes, decl=False):
        self.sim = sim
        self.parent_block = None
        self.array = array
        self.array_type = a_type
        self.decl = decl
        self.prim_size = SizeofAST(sim, a_type)
        self.size = reduce(operator.mul, sizes) * self.prim_size
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


class ReallocAST:
    def __init__(self, sim, array, size):
        self.sim = sim
        self.parent_block = None
        self.array = array
        self.size = size
        self.sim.add_statement(self)

    def children(self):
        return [self.array, self.size]

    def generate(self, mem=False):
        self.sim.code_gen.generate_realloc(
            self.array.generate(), self.size.generate())

    def transform(self, fn):
        self.array = self.array.transform(fn)
        self.size = self.size.transform(fn)
        return fn(self)
