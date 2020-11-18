from ast.lit import as_lit_ast


class Select:
    def __init__(self, sim, cond, expr_if, expr_else):
        self.sim = sim
        self.cond = as_lit_ast(sim, cond)
        self.expr_if = as_lit_ast(expr_if)
        self.expr_else = as_lit_ast(expr_else)

    def children(self):
        return [self.cond, self.expr_if, self.expr_else]

    def generate(self):
        cond_gen = self.cond.generate()
        expr_if_gen = self.expr_if.generate_inline()
        expr_else_gen = self.expr_else.generate_inline()
        self.sim.code_gen.generate_select(cond_gen, expr_if_gen, expr_else_gen)

    def transform(self, fn):
        self.cond = self.cond.transform(fn)
        self.expr_if = self.expr_if.transform(fn)
        self.expr_else = self.expr_else.transform(fn)
        return fn(self)

