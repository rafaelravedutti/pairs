from ast.assign import Assign
from ast.data_types import Type_Array
from ast.expr import Expr
from ast.lit import as_lit_ast
from ast.memory import Realloc
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
        self.arr_sizes = \
            [arr_sizes] if not isinstance(arr_sizes, list) \
            else [as_lit_ast(sim, s) for s in arr_sizes]
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

    def scope(self):
        return self.sim.global_scope

    def ndims(self):
        return self.arr_ndims

    def alloc_size(self):
        return reduce((lambda x, y: x * y), [s for s in self.arr_sizes])

    def realloc(self):
        return Realloc(self.sim, self, self.alloc_size())

    def children(self):
        return []

    def generate(self, mem=False):
        return self.arr_name

    def transform(self, fn):
        return fn(self)


class ArrayAccess:
    last_acc = 0

    def new_id():
        ArrayAccess.last_acc += 1
        return ArrayAccess.last_acc - 1

    def __init__(self, sim, array, index):
        self.sim = sim
        self.acc_id = ArrayAccess.new_id()
        self.array = array
        self.indexes = [as_lit_ast(sim, index)]
        self.index = None
        self.generated = False
        self.check_and_set_index()

    def __str__(self):
        return f"ArrayAccess<array: {self.array}, indexes: {self.indexes}>"

    def __add__(self, other):
        return Expr(self.sim, self, other, '+')

    def __mul__(self, other):
        return Expr(self.sim, self, other, '*')

    def __rmul__(self, other):
        return Expr(self.sim, other, self, '*')

    def __getitem__(self, index):
        assert self.index is None, \
            "Number of indexes higher than array dimension!"
        self.indexes.append(as_lit_ast(self.sim, index))
        self.check_and_set_index()
        return self

    def check_and_set_index(self):
        if len(self.indexes) == self.array.ndims():
            sizes = self.array.sizes()
            for s in range(0, len(sizes)):
                self.index = (self.indexes[s] if self.index is None
                              else self.index * sizes[s] + self.indexes[s])

            self.index = as_lit_ast(self.sim, self.index)

    def set(self, other):
        return self.sim.add_statement(Assign(self.sim, self, other))

    def add(self, other):
        return self.sim.add_statement(Assign(self.sim, self, self + other))

    def type(self):
        return self.array.type() if self.index is None else Type_Array

    def scope(self):
        if self.index is None:
            scope = None
            for i in self.indexes:
                iscp = i.scope()
                if scope is None or iscp > scope:
                    scope = iscp

            return scope

        return self.index.scope()

    def children(self):
        return [self.array] + self.indexes

    def generate(self, mem=False):
        agen = self.array.generate()
        igen = self.index.generate()
        if mem is False and self.generated is False:
            self.sim.code_gen.generate_array_access(
                self.acc_id, self.array.type(), agen, igen)
            self.generated = True

        return self.sim.code_gen.generate_array_access_ref(
            self.acc_id, agen, igen, mem)

    def transform(self, fn):
        self.array = self.array.transform(fn)
        self.indexes = [i.transform(fn) for i in self.indexes]
        return fn(self)
