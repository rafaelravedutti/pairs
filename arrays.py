from assign import AssignAST
from data_types import Type_Array
from expr import ExprAST
from lit import is_literal, LitAST

class ArrayND:
    def __init__(self, sim, arr_name, arr_sizes, arr_type):
        self.sim = sim
        self.arr_name = arr_name
        self.arr_sizes = [arr_sizes] if not isinstance(arr_sizes, list) else [LitAST(s) if is_literal(s) else s for s in arr_sizes]
        self.arr_type = arr_type
        self.arr_ndims = len(self.arr_sizes)

    def __str__(self):
        return f"ArrayND<name: {self.arr_name}, sizes: {self.arr_sizes}, type: {self.arr_type}>"

    def name(self):
        return self.arr_name

    def sizes(self):
        return self.arr_sizes

    def type(self):
        return self.arr_type

    def ndims(self):
        return self.ndims

    def __getitem__(self, expr_ast):
        return ArrayAccess(self.sim, self, expr_ast)

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

    def __rmul__(self, other):
        return ExprAST(self.sim, other, self, '*')

    def __getitem__(self, expr_ast):
        self.indexes.append(expr_ast)
        return self

    def set(self, other):
        return self.sim.capture_statement(AssignAST(self.sim, self, other))

    def add(self, other):
        return self.sim.capture_statement(AssignAST(self.sim, self, self + other))

    def type(self):
        return self.array.type() if len(self.indexes) == self.array.ndims() else Type_Array

    def generate(self, mem=False):
        index = None
        sizes = self.array.sizes()
        for s in range(0, len(sizes)): 
            index = self.indexes[s] if index is None else index * sizes[s] + self.indexes[s]

        index = LitAST(index) if is_literal(index) else index
        return f"{self.array.generate()}[{index.generate()}]"

    def transform(self, fn):
        self.array = self.array.transform(fn)
        return fn(self)
