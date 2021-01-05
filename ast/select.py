from ast.expr import BinOp
from ast.lit import as_lit_ast


class Select:
    def __init__(self, sim, cond, expr_if, expr_else):
        self.sim = sim
        self.cond = as_lit_ast(sim, cond)
        self.expr_if = BinOp.inline(as_lit_ast(sim, expr_if))
        self.expr_else = BinOp.inline(as_lit_ast(sim, expr_else))

    def children(self):
        return [self.cond, self.expr_if, self.expr_else]

    def transform(self, fn):
        self.cond = self.cond.transform(fn)
        self.expr_if = self.expr_if.transform(fn)
        self.expr_else = self.expr_else.transform(fn)
        return fn(self)

