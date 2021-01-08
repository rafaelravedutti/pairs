from ast.ast_node import ASTNode
from ast.data_types import Type_Int, Type_Float


class Cast(ASTNode):
    def __init__(self, sim, expr, cast_type):
        super().__init__(sim)
        self.expr = expr
        self.cast_type = cast_type

    def __str__(self):
        return f"Cast<expr: {self.expr}, cast_type: {self.cast_type}>"

    def int(sim, expr):
        return Cast(sim, expr, Type_Int)

    def float(sim, expr):
        return Cast(sim, expr, Type_Float)

    def type(self):
        return self.cast_type

    def is_mutable(self):
        return self.expr.is_mutable()

    def scope(self):
        return self.expr.scope()

    def children(self):
        return [self.expr]

    def transform(self, fn):
        self.expr = self.expr.transform(fn)
        return fn(self)
