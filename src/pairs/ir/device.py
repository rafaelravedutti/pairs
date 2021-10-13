from pairs.ir.ast_node import ASTNode


class DeviceCopy(ASTNode):
    def __init__(self, sim, prop):
        super().__init__(sim)
        self.prop = prop

    def children(self):
        return [self.prop]
