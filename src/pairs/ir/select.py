from pairs.ir.ast_node import ASTNode
from pairs.ir.bin_op import BinOp
from pairs.ir.lit import Lit


class Select(ASTNode):
    def __init__(self, sim, cond, expr_if, expr_else):
        super().__init__(sim)
        self.cond = Lit.cvt(sim, cond)
        self.expr_if = BinOp.inline(Lit.cvt(sim, expr_if))
        self.expr_else = BinOp.inline(Lit.cvt(sim, expr_else))

    def children(self):
        return [self.cond, self.expr_if, self.expr_else]
