from ast.data_types import Type_Int, Type_Float


class Sqrt:
    def __init__(self, sim, expr, cast_type):
        self.sim = sim
        self.expr = expr

    def __str__(self):
        return f"Sqrt<expr: {self.expr}>"

    def type(self):
        return self.expr.type()

    def scope(self):
        return self.expr.scope()

    def children(self):
        return [self.expr]

    def generate(self, mem=False):
        return self.sim.code_gen.generate_sqrt(self.expr.generate())

    def transform(self, fn):
        self.expr = self.expr.transform(fn)
        return fn(self)
