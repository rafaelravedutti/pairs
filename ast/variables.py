from ast.assign import Assign
from ast.expr import BinOp


class Variables:
    def __init__(self, sim):
        self.sim = sim
        self.vars = []
        self.nvars = 0

    def add(self, v_name, v_type, v_value=0):
        v = Var(self.sim, v_name, v_type, v_value)
        self.vars.append(v)
        return v

    def all(self):
        return self.vars

    def find(self, v_name):
        var = [v for v in self.vars if v.name() == v_name]
        if var:
            return var[0]

        return None

class Var:
    def __init__(self, sim, var_name, var_type, init_value=0):
        self.sim = sim
        self.var_name = var_name
        self.var_type = var_type
        self.var_init_value = init_value
        self.mutable = True
        self.var_bonded_arrays = []

    def __str__(self):
        return f"Var<name: {self.var_name}, type: {self.var_type}>"

    def __add__(self, other):
        return BinOp(self.sim, self, other, '+')

    def __radd__(self, other):
        return BinOp(self.sim, other, self, '+')

    def __sub__(self, other):
        return BinOp(self.sim, self, other, '-')

    def __mul__(self, other):
        return BinOp(self.sim, self, other, '*')

    def __rmul__(self, other):
        return BinOp(self.sim, other, self, '*')

    def __truediv__(self, other):
        return BinOp(self.sim, self, other, '/')

    def __rtruediv__(self, other):
        return BinOp(self.sim, other, self, '/')

    def __lt__(self, other):
        return BinOp(self.sim, self, other, '<')

    def __le__(self, other):
        return BinOp(self.sim, self, other, '<=')

    def __gt__(self, other):
        return BinOp(self.sim, self, other, '>')

    def __ge__(self, other):
        return BinOp(self.sim, self, other, '>=')

    def inv(self):
        return BinOp(self.sim, 1.0, self, '/')

    def set(self, other):
        return self.sim.add_statement(Assign(self.sim, self, other))

    def add(self, other):
        return self.sim.add_statement(Assign(self.sim, self, self + other))

    def name(self):
        return self.var_name

    def type(self):
        return self.var_type

    def set_initial_value(self, value):
        self.var_init_value = value

    def init_value(self):
        return self.var_init_value

    def add_bonded_array(self, array):
        self.var_bonded_arrays.append(array)

    def bonded_arrays(self):
        return self.var_bonded_arrays

    def is_mutable(self):
        return self.mutable

    def scope(self):
        return self.sim.global_scope

    def children(self):
        return []

    def transform(self, fn):
        return fn(self)


class VarDecl:
    def __init__(self, sim, var):
        self.sim = sim
        self.var = var
        self.sim.add_statement(self)

    def children(self):
        return []

    def transform(self, fn):
        return fn(self)
