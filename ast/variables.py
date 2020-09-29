from ast.assign import AssignAST
from ast.expr import ExprAST


class Variables:
    def __init__(self, sim):
        self.sim = sim
        self.vars = []
        self.nvars = 0

    def add(self, v_name, v_type):
        v = Var(self.sim, v_name, v_type)
        self.vars.append(v)
        return v

    def find(self, v_name):
        return [v for v in self.vars if v.name() == v_name][0]


class Var:
    def __init__(self, sim, var_name, var_type):
        self.sim = sim
        self.var_name = var_name
        self.var_type = var_type

    def __str__(self):
        return f"Var<name: {self.var_name}, type: {self.var_type}>"

    def __add__(self, other):
        return ExprAST(self.sim, self, other, '+')

    def __radd__(self, other):
        return ExprAST(self.sim, other, self, '+')

    def __sub__(self, other):
        return ExprAST(self.sim, self, other, '-')

    def __mul__(self, other):
        return ExprAST(self.sim, self, other, '*')

    def __rmul__(self, other):
        return ExprAST(self.sim, other, self, '*')

    def __truediv__(self, other):
        return ExprAST(self.sim, self, other, '/')

    def __rtruediv__(self, other):
        return ExprAST(self.sim, other, self, '/')

    def __lt__(self, other):
        return ExprAST(self.sim, self, other, '<')

    def __gt__(self, other):
        return ExprAST(self.sim, self, other, '>')

    def inv(self):
        return ExprAST(self.sim, 1.0, self, '/')

    def set(self, other):
        return self.sim.capture_statement(AssignAST(self.sim, self, other))

    def add(self, other):
        return self.sim.capture_statement(
            AssignAST(self.sim, self, self + other))

    def name(self):
        return self.var_name

    def type(self):
        return self.var_type

    def generate(self, mem=False):
        return self.var_name

    def transform(self, fn):
        return fn(self)
