from functools import reduce
from pairs.ir.assign import Assign
from pairs.ir.ast_node import ASTNode
from pairs.ir.bin_op import BinOp, ASTTerm
from pairs.ir.layouts import Layouts
from pairs.ir.lit import Lit
from pairs.ir.memory import Realloc
from pairs.ir.sizeof import Sizeof
from pairs.ir.variables import Var


class Arrays:
    def __init__(self, sim):
        self.sim = sim
        self.arrays = []

    def add(self, a_name, a_sizes, a_type, a_layout=Layouts.AoS):
        array = ArrayND(self.sim, a_name, a_sizes, a_type, a_layout)
        self.arrays.append(array)
        return array

    def add_static(self, a_name, a_sizes, a_type, a_layout=Layouts.AoS, init_value=None):
        array = ArrayStatic(self.sim, a_name, a_sizes, a_type, a_layout, init_value)
        self.arrays.append(array)
        return array

    def all(self):
        return self.arrays

    def statics(self):
        return [a for a in self.arrays if a.is_static()]

    def narrays(self):
        return len(self.arrays)

    def find(self, a_name):
        array = [a for a in self.arrays if a.name() == a_name]
        if array:
            return array[0]

        return None


class Array(ASTNode):
    last_array_id = 0

    def __init__(self, sim, a_name, a_sizes, a_type, a_layout=Layouts.AoS):
        super().__init__(sim)
        self.arr_id = Array.last_array_id
        self.arr_name = a_name
        self.arr_sizes = [Lit.cvt(sim, a_sizes)] if not isinstance(a_sizes, list) \
                         else [Lit.cvt(sim, s) for s in a_sizes]
        self.arr_type = a_type
        self.arr_layout = a_layout
        self.arr_ndims = len(self.arr_sizes)
        self.static = False
        self.device_flag = False
        Array.last_array_id += 1

        for var in [s for s in self.arr_sizes if isinstance(s, Var)]:
            var.add_bonded_array(self)

    def __getitem__(self, expr_ast):
        return ArrayAccess(self.sim, self, expr_ast)

    def id(self):
        return self.arr_id

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
    def __init__(self, sim, a_name, a_sizes, a_type, a_layout=Layouts.AoS, init_value=None):
        super().__init__(sim, a_name, a_sizes, a_type, a_layout)
        self.init_value = init_value
        self.static = True

    def __str__(self):
        return f"ArrayStatic<{self.arr_name}>"

    def realloc(self):
        raise Exception("Static array cannot be reallocated!")

    def update(self):
        raise Exception("Static array cannot be updated!")


class ArrayND(Array):
    def __init__(self, sim, a_name, a_sizes, a_type, a_layout=Layouts.AoS):
        super().__init__(sim, a_name, a_sizes, a_type, a_layout)

    def __str__(self):
        return f"ArrayND<{self.arr_name}>"

    def realloc(self):
        return Realloc(self.sim, self, self.alloc_size())

    def update(self):
        return UpdateArray(self.sim, self, self.alloc_size())


class ArrayAccess(ASTTerm):
    last_acc = 0

    def new_id():
        ArrayAccess.last_acc += 1
        return ArrayAccess.last_acc - 1

    def __init__(self, sim, array, index):
        super().__init__(sim)
        self.acc_id = ArrayAccess.new_id()
        self.array = array
        self.indexes = [Lit.cvt(sim, index)]
        self.index = None
        self.inlined = False
        self.terminals = set()
        self.check_and_set_index()

    def __str__(self):
        return f"ArrayAccess<{self.array}, {self.indexes}>"

    def __getitem__(self, index):
        assert self.index is None, "Number of indexes higher than array dimension!"
        self.indexes.append(Lit.cvt(self.sim, index))
        self.check_and_set_index()
        return self

    def inline_rec(self):
        self.inlined = True
        return self

    def check_and_set_index(self):
        if len(self.indexes) == self.array.ndims():
            sizes = self.array.sizes()
            layout = self.array.layout()

            if layout == Layouts.AoS:
                for s in range(0, len(sizes)):
                    self.index = (self.indexes[s] if self.index is None
                                  else self.index * sizes[s] + self.indexes[s])

            elif layout == Layouts.SoA:
                for s in reversed(range(0, len(sizes))):
                    self.index = (self.indexes[s] if self.index is None
                                  else self.index * sizes[s] + self.indexes[s])

            else:
                raise Exception("Invalid data layout!")

            self.index = Lit.cvt(self.sim, self.index)

    def set(self, other):
        return self.sim.add_statement(Assign(self.sim, self, other))

    def add(self, other):
        return self.sim.add_statement(Assign(self.sim, self, self + other))

    def id(self):
        return self.acc_id

    def type(self):
        return self.array.type()
        # return self.array.type() if self.index is None else Types.Array

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        if self.index is not None:
            return [self.array, self.index]

        return [self.array] + self.indexes


class ArrayDecl(ASTNode):
    def __init__(self, sim, array):
        super().__init__(sim)
        self.array = array
        self.sim.add_statement(self)


class RegisterArray(ASTNode):
    def __init__(self, sim, array, size):
        super().__init__(sim)
        self._array = array
        self._prim_size = Sizeof(sim, array.type())
        self._size = BinOp.inline(self._prim_size * size)
        self.sim.add_statement(self)

    def array(self):
        return self._array

    def size(self):
        return self._size

    def __str__(self):
        return f"RegisterArray<{self._array.name()}>"


class UpdateArray(ASTNode):
    def __init__(self, sim, array, size):
        super().__init__(sim)
        self._array = array
        self._prim_size = Sizeof(sim, array.type())
        self._size = BinOp.inline(self._prim_size * size)
        self.sim.add_statement(self)

    def array(self):
        return self._array

    def size(self):
        return self._size

    def __str__(self):
        return f"UpdateArray<{self._array.name()}>"
