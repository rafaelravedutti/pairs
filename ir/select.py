from ir.ast_node import ASTNode
from ir.bin_op import BinOp
from ir.lit import as_lit_ast


class Select(ASTNode):
    def __init__(self, sim, cond, expr_if, expr_else):
        super().__init__(sim)
        self.cond = as_lit_ast(sim, cond)
        self.expr_if = BinOp.inline(as_lit_ast(sim, expr_if))
        self.expr_else = BinOp.inline(as_lit_ast(sim, expr_else))

    def children(self):
        return [self.cond, self.expr_if, self.expr_else]