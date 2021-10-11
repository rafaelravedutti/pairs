from pairs.ir.bin_op import ASTTerm
from pairs.ir.data_types import Type_Int, Type_Float


class Sqrt(ASTTerm):
    def __init__(self, sim, expr, cast_type):
        super().__init__(sim)
        self.expr = expr

    def __str__(self):
        return f"Sqrt<expr: {self.expr}>"

    def type(self):
        return self.expr.type()

    def scope(self):
        return self.expr.scope()

    def children(self):
        return [self.expr]


class Ceil(ASTTerm):
    def __init__(self, sim, expr):
        assert expr.type() == Type_Float, "Expression must be of floating-point type!"
        super().__init__(sim)
        self.expr = expr

    def __str__(self):
        return f"Ceil<expr: {self.expr}>"

    def type(self):
        return Type_Int

    def scope(self):
        return self.expr.scope()

    def children(self):
        return [self.expr]
