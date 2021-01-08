from ast.ast_node import ASTNode


class Print(ASTNode):
    def __init__(self, sim, string):
        super().__init__(sim)
        self.string = string

    def __str__(self):
        return f"Print<{self.string}>"
