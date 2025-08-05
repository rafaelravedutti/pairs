from pairs.ir.ast_term import ASTTerm 
from pairs.ir.operator_class import OperatorClass


class Parameter(ASTTerm):
    def __init__(self, sim, param_name, param_type, order=None):
        super().__init__(sim, OperatorClass.from_type(param_type))
        self.param_name = param_name
        self.param_type = param_type
        self.order = order

    def __str__(self):
        return f"Parameter<{self.param_name}>"

    def name(self):
        return self.param_name

    def type(self):
        return self.param_type
