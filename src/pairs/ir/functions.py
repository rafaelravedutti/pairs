from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.lit import Lit
from pairs.ir.types import Types


class Call(ASTTerm):
    def __init__(self, sim, func_name, params, return_type):
        super().__init__(sim, ScalarOp)
        self.func_name = func_name
        self.params = [Lit.cvt(sim, p) for p in params]
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
        super().__init__(sim, func_name, parameters, Types.Int32)


class Call_Void(Call):
    def __init__(self, sim, func_name, parameters):
        super().__init__(sim, func_name, parameters, Types.Invalid)
        sim.add_statement(self)
