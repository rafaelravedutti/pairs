from ast.assign import Assign
from ast.ast_node import ASTNode
from ast.bin_op import BinOp, ASTTerm
from ast.data_types import Type_Array
from ast.layouts import Layout_AoS, Layout_SoA
from ast.lit import as_lit_ast
from ast.memory import Realloc
from ast.variables import Var
from functools import reduce


class Arrays:
    def __init__(self, sim):
        self.sim = sim
        self.arrays = []
        self.narrays = 0

    def add(self, a_name, a_sizes, a_type, a_layout=Layout_AoS):
        array = ArrayND(self.sim, a_name, a_sizes, a_type, a_layout)
        self.arrays.append(array)
        return array

    def add_static(self, a_name, a_sizes, a_type, a_layout=Layout_AoS):
        array = ArrayStatic(self.sim, a_name, a_sizes, a_type, a_layout)
        self.arrays.append(array)
        return array

    def all(self):
        return self.arrays

    def find(self, a_name):
        array = [a for a in self.arrays if a.name() == a_name]
        if array:
            return array[0]

        return None


class Array(ASTNode):
    def __init__(self, sim, a_name, a_sizes, a_type, a_layout=Layout_AoS):
        super().__init__(sim)
        self.arr_name = a_name
        self.arr_sizes = \
            [as_lit_ast(sim, a_sizes)] if not isinstance(a_sizes, list) \
            else [as_lit_ast(sim, s) for s in a_sizes]
        self.arr_type = a_type
        self.arr_layout = a_layout
        self.arr_ndims = len(self.arr_sizes)
        self.static = False
        for var in [s for s in self.arr_sizes if isinstance(s, Var)]:
            var.add_bonded_array(self)


    def __getitem__(self, expr_ast):
        return ArrayAccess(self.sim, self, expr_ast)

    def name(self):
        return self.arr_name

    def sizes(self):
        return self.arr_sizes

    def type(self):
        return self.arr_type

    def layout(self):
        return self.arr_layout

    def ndims(self):
        return self.arr_ndims

    def is_static(self):
        return self.static

    def alloc_size(self):
        return reduce((lambda x, y: x * y), [s for s in self.arr_sizes])


class ArrayStatic(Array):
    def __init__(self, sim, a_name, a_sizes, a_type, a_layout=Layout_AoS):
        super().__init__(sim, a_name, a_sizes, a_type, a_layout)
        self.static = True

    def __str__(self):
        return (f"ArrayStatic<name: {self.arr_name}, " +
                f"sizes: {self.arr_sizes}, " +
                f"type: {self.arr_type}>")

    def realloc(self):
        raise Exception("Static array cannot be reallocated!")


class ArrayND(Array):
    def __init__(self, sim, a_name, a_sizes, a_type, a_layout=Layout_AoS):
        super().__init__(sim, a_name, a_sizes, a_type, a_layout)

    def __str__(self):
        return (f"ArrayND<name: {self.arr_name}, sizes: {self.arr_sizes}, " +
                f"type: {self.arr_type}>")

    def realloc(self):
        return Realloc(self.sim, self, self.alloc_size())


class ArrayAccess(ASTTerm):
    last_acc = 0

    def new_id():
        ArrayAccess.last_acc += 1
        return ArrayAccess.last_acc - 1

    def __init__(self, sim, array, index):
        super().__init__(sim)
        self.acc_id = ArrayAccess.new_id()
        self.array = array
        self.indexes = [as_lit_ast(sim, index)]
        self.index = None
        self.mutable = True
        self.generated = False
        self.check_and_set_index()

    def __str__(self):
        return f"ArrayAccess<array: {self.array}, indexes: {self.indexes}>"

    def __getitem__(self, index):
        assert self.index is None, "Number of indexes higher than array dimension!"
        self.indexes.append(as_lit_ast(self.sim, index))
        self.check_and_set_index()
        return self

    def check_and_set_index(self):
        if len(self.indexes) == self.array.ndims():
            sizes = self.array.sizes()
            layout = self.array.layout()

            if layout == Layout_AoS:
                for s in range(0, len(sizes)):
                    self.index = (self.indexes[s] if self.index is None
                                  else self.index * sizes[s] + self.indexes[s])

            elif layout == Layout_SoA:
                for s in reversed(range(0, len(sizes))):
                    self.index = (self.indexes[s] if self.index is None
                                  else self.index * sizes[s] + self.indexes[s])

            else:
                raise Exception("Invalid data layout!")

            self.index = as_lit_ast(self.sim, self.index)

    def set(self, other):
        return self.sim.add_statement(Assign(self.sim, self, other))

    def add(self, other):
        return self.sim.add_statement(Assign(self.sim, self, self + other))

    def id(self):
        return self.acc_id

    def type(self):
        return self.array.type()
        # return self.array.type() if self.index is None else Type_Array

    def is_mutable(self):
        return self.mutable

    def scope(self):
        if self.index is None:
            scope = None
            for i in self.indexes:
                index_scp = i.scope()
                if scope is None or index_scp > scope:
                    scope = index_scp

            return scope

        return self.index.scope()

    def children(self):
        return [self.array] + self.indexes

    def transform(self, fn):
        self.array = self.array.transform(fn)
        self.indexes = [i.transform(fn) for i in self.indexes]

        if self.index is not None:
            self.index = self.index.transform(fn)

        return fn(self)


class ArrayDecl(ASTNode):
    def __init__(self, sim, array):
        super().__init__(sim)
        self.array = array
        self.sim.add_statement(self)
