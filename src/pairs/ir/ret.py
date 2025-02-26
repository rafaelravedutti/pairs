from pairs.ir.ast_node import ASTNode

class Return(ASTNode):
    def __init__(self, sim, expr):
        super().__init__(sim)
        self.expr = expr
        self.sim.add_statement(self)

    def __str__(self):
        return f"Return<{self.expr}>"

    def children(self):
        return [self.expr]