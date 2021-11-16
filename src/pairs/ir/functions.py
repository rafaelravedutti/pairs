from pairs.ir.bin_op import ASTTerm
from pairs.ir.data_types import Type_Int, Type_Invalid
from pairs.ir.lit import as_lit_ast


class Call(ASTTerm):
    def __init__(self, sim, func_name, params, return_type):
        super().__init__(sim)
        self.func_name = func_name
        self.params = [as_lit_ast(sim, p) for p in params]
        self.return_type = return_type

    def name(self):
        return self.func_name

    def parameters(self):
        return self.params

    def type(self):
        return self.return_type

    def children(self):
        return self.params


class Call_Int(Call):
    def __init__(self, sim, func_name, parameters):
        super().__init__(sim, func_name, parameters, Type_Int)


class Call_Void(Call):
    def __init__(self, sim, func_name, parameters):
        super().__init__(sim, func_name, parameters, Type_Invalid)
        sim.add_statement(self)
