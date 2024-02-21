from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm 
from pairs.ir.assign import Assign
from pairs.ir.scalars import ScalarOp
from pairs.ir.lit import Lit
from pairs.ir.operator_class import OperatorClass


class Variables:
    temp_id = 0

    def new_temp_id():
        Variables.temp_id += 1
        return Variables.temp_id - 1

    def __init__(self, sim):
        self.sim = sim
        self.vars = []
        self.nvars = 0

    def add(self, v_name, v_type, v_value=0, v_runtime_track=False):
        var = Var(self.sim, v_name, v_type, v_value, v_runtime_track)
        self.vars.append(var)
        return var

    def add_temp(self, init):
        lit = Lit.cvt(self.sim, init)
        tmp_id = Variables.new_temp_id()
        tmp_var = Var(self.sim, f"tmp{tmp_id}", lit.type(), temp=True)
        Assign(self.sim, tmp_var, lit)
        return tmp_var

    def all(self):
        return self.vars

    def find(self, v_name):
        var = [v for v in self.vars if v.name() == v_name]
        return var[0] if var else None


class Var(ASTTerm):
    def __init__(self, sim, var_name, var_type, init_value=0, runtime_track=False, temp=False):
        super().__init__(sim, OperatorClass.from_type(var_type))
        self.var_name = var_name
        self.var_type = var_type
        self.var_init_value = Lit.cvt(sim, init_value)
        self.var_runtime_track = runtime_track
        self.var_temporary = temp
        self.mutable = True
        self.var_bonded_arrays = []
        self.device_flag = False

        if temp:
            DeclareVariable(sim, self)

    def __str__(self):
        return f"Var<{self.var_name}>"

    def copy(self, deep=False):
        # Terminal copies are just themselves
        return self

    def name(self):
        return self.var_name

    def type(self):
        return self.var_type

    def temporary(self):
        return self.var_temporary

    def set_initial_value(self, value):
        self.var_init_value = value

    def init_value(self):
        return self.var_init_value

    def runtime_track(self):
        return self.var_runtime_track

    def add_bonded_array(self, array):
        self.var_bonded_arrays.append(array)

    def bonded_arrays(self):
        return self.var_bonded_arrays


class DeclareVariable(ASTNode):
    def __init__(self, sim, var):
        super().__init__(sim)
        self.var = var
        self.sim.add_statement(self)


class Deref(ASTTerm):
    def __init__(self, sim, var):
        super().__init__(sim, ScalarOp)
        self._var = var

    def __str__(self):
        return f"Deref<{self.var.name()}>"

    @property
    def var(self):
        return self._var

    def type(self):
        return self._var.type()

    def children(self):
        return [self._var]
