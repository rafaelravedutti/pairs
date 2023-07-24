from pairs.ir.ast_node import ASTNode
from pairs.ir.bin_op import ASTTerm, BinOp, VectorAccess
from pairs.ir.lit import Lit
from pairs.ir.vector_expr import VectorExpression


class Select(ASTTerm, VectorExpression):
    last_select = 0

    def new_id():
        Select.last_select += 1
        return Select.last_select - 1

    def __init__(self, sim, cond, expr_if, expr_else):
        super().__init__(sim)
        self.select_id = Select.new_id()
        self.cond = Lit.cvt(sim, cond)
        self.expr_if = BinOp.inline(Lit.cvt(sim, expr_if))
        self.expr_else = BinOp.inline(Lit.cvt(sim, expr_else))
        assert self.expr_if.type() == self.expr_else.type(), "Select: expressions must be of same type!"

    def __str__(self):
        return f"Select<{self.cond}, {self.expr_if}, {self.expr_else}>"

    def type(self):
        return self.expr_if.type()

    def inline_rec(self):
        self.inlined = True
        return self

    def children(self):
        return [self.cond, self.expr_if, self.expr_else]

    def __getitem__(self, index):
        super().__getitem__(index)
        return VectorAccess(self.sim, self, Lit.cvt(self.sim, index))
