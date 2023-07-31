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
        #self.expr_if = BinOp.inline(Lit.cvt(sim, expr_if))
        #self.expr_else = BinOp.inline(Lit.cvt(sim, expr_else))
        self.expr_if = Lit.cvt(sim, expr_if)
        self.expr_else = Lit.cvt(sim, expr_else)
        self.inlined = False
        assert self.expr_if.type() == self.expr_else.type(), "Select: expressions must be of same type!"

    def __str__(self):
        return f"Select<{self.cond}, {self.expr_if}, {self.expr_else}>"

    def id(self):
        return self.select_id

    def type(self):
        return self.expr_if.type()

    def inline_recursively(self):
        method_name = "inline_recursively"
        self.inlined = True

        if hasattr(self.cond, method_name) and callable(getattr(self.cond, method_name)):
            self.cond.inline_recursively()

        if hasattr(self.expr_if, method_name) and callable(getattr(self.expr_if, method_name)):
            self.expr_if.inline_recursively()

        if hasattr(self.expr_else, method_name) and callable(getattr(self.expr_else, method_name)):
            self.expr_else.inline_recursively()

        return self

    def children(self):
        return [self.cond, self.expr_if, self.expr_else]

    def __getitem__(self, index):
        super().__getitem__(index)
        return VectorAccess(self.sim, self, Lit.cvt(self.sim, index))
