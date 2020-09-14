from data_types import Type_Int, Type_Float

class CastAST:
    def __init__(self, expr, cast_type):
        self.expr = expr
        self.cast_type = cast_type

    def int(expr):
        return CastAST(expr, Type_Int)

    def type(self):
        return self.cast_type

    def generate(self, mem=False):
        t = 'double' if self.cast_type == Type_Float else 'int' if self.cast_type == Type_Int else 'bool'
        return f"({t})({self.expr.generate()})"

    def transform(self, fn):
        self.expr = self.expr.transform(fn)
        return fn(self)
