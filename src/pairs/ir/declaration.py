from pairs.ir.ast_node import ASTNode


class Decl(ASTNode):
    def __init__(self, sim, elem):
        super().__init__(sim)
        self.elem = elem

    def __str__(self):
        return f"Decl<self.elem>"

    def children(self):
        return [self.elem]
