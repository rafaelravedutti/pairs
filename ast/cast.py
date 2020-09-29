from ast.data_types import Type_Int, Type_Float

class CastAST:
    def __init__(self, sim, expr, cast_type):
        self.sim = sim
        self.expr = expr
        self.cast_type = cast_type

    def __str__(self):
        return f"Cast<expr: {self.expr}, cast_type: {self.cast_type}>"

    def int(sim, expr):
        return CastAST(sim, expr, Type_Int)

    def float(sim, expr):
        return CastAST(sim, expr, Type_Float)

    def type(self):
        return self.cast_type

    def generate(self, mem=False):
        self.sim.code_gen.generate_cast(self.cast_type, self.expr.generate())

    def transform(self, fn):
        self.expr = self.expr.transform(fn)
        return fn(self)
