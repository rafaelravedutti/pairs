from ast.assign import AssignAST
from ast.data_types import Type_Array
from ast.expr import ExprAST
from ast.lit import is_literal, LitAST
from ast.memory import ReallocAST
from functools import reduce


class Arrays:
    def __init__(self, sim):
        self.sim = sim
        self.arrays = []
        self.narrays = 0

    def add(self, a_name, a_sizes, a_type):
        a = ArrayND(self.sim, a_name, a_sizes, a_type)
        self.arrays.append(a)
        return a

    def find(self, a_name):
        return [a for a in self.arrays if a.name() == a_name][0]


class ArrayND:
    def __init__(self, sim, arr_name, arr_sizes, arr_type):
        self.sim = sim
        self.arr_name = arr_name
        self.arr_sizes = ([arr_sizes] if not isinstance(arr_sizes, list)
                          else [LitAST(s) if is_literal(s) else s
                                for s in arr_sizes])
        self.arr_type = arr_type
        self.arr_ndims = len(self.arr_sizes)

    def __str__(self):
        return (f"ArrayND<name: {self.arr_name}, sizes: {self.arr_sizes}, " +
                f"type: {self.arr_type}>")

    def __getitem__(self, expr_ast):
        return ArrayAccess(self.sim, self, expr_ast)

    def name(self):
        return self.arr_name

    def sizes(self):
        return self.arr_sizes

    def type(self):
        return self.arr_type

    def ndims(self):
        return self.arr_ndims

    def alloc_size(self):
        return reduce((lambda x, y: x * y), [s for s in self.arr_sizes])

    def realloc(self):
        return ReallocAST(self.sim, self, self.alloc_size())

    def generate(self, mem=False):
        return self.arr_name

    def transform(self, fn):
        return fn(self)


class ArrayAccess:
    def __init__(self, sim, array, index):
        self.sim = sim
        self.array = array
        self.indexes = [index]

    def __str__(self):
        return f"ArrayAccess<array: {self.array}, indexes: {self.indexes}>"

    def __add__(self, other):
        return ExprAST(self.sim, self, other, '+')

    def __mul__(self, other):
        return ExprAST(self.sim, self, other, '*')

    def __rmul__(self, other):
        return ExprAST(self.sim, other, self, '*')

    def __getitem__(self, expr_ast):
        self.indexes.append(expr_ast)
        return self

    def set(self, other):
        return self.sim.add_statement(AssignAST(self.sim, self, other))

    def add(self, other):
        return self.sim.add_statement(AssignAST(self.sim, self, self + other))

    def type(self):
        return (self.array.type() if len(self.indexes) == self.array.ndims()
                else Type_Array)

    def generate(self, mem=False):
        index = None
        sizes = self.array.sizes()
        for s in range(0, len(sizes)):
            index = (self.indexes[s] if index is None
                     else index * sizes[s] + self.indexes[s])

        index = LitAST(index) if is_literal(index) else index
        return self.sim.code_gen.generate_array_access(
            self.array.generate(), index.generate())

    def transform(self, fn):
        self.array = self.array.transform(fn)
        return fn(self)
