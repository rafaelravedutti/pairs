from pairs.ir.bin_op import ASTTerm
from pairs.ir.types import Types


class Sqrt(ASTTerm):
    def __init__(self, sim, expr, cast_type):
        super().__init__(sim)
        self.expr = expr

    def __str__(self):
        return f"Sqrt<{self.expr}>"

    def type(self):
        return self.expr.type()

    def children(self):
        return [self.expr]


class Ceil(ASTTerm):
    def __init__(self, sim, expr):
        assert Types.is_real(expr.type()), "Expression must be of real type!"
        super().__init__(sim)
        self.expr = expr

    def __str__(self):
        return f"Ceil<{self.expr}>"

    def type(self):
        return Types.Int32

    def children(self):
        return [self.expr]
