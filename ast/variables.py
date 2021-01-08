from ast.ast_node import ASTNode
from ast.assign import Assign
from ast.bin_op import ASTTerm 


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

class Var(ASTTerm):
    def __init__(self, sim, var_name, var_type, init_value=0):
        super().__init__(sim)
        self.var_name = var_name
        self.var_type = var_type
        self.var_init_value = init_value
        self.mutable = True
        self.var_bonded_arrays = []

    def __str__(self):
        return f"Var<name: {self.var_name}, type: {self.var_type}>"

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


class VarDecl(ASTNode):
    def __init__(self, sim, var):
        super().__init__(sim)
        self.var = var
        self.sim.add_statement(self)
