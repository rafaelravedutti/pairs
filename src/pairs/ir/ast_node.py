from pairs.ir.types import Types


class ASTNode:
    def __init__(self, sim):
        self.sim = sim
        self.parent_block = None # Set during SetParentBlock transformation

    def __str__(self):
        return "ASTNode<>"

    def type(self):
        return Types.Invalid

    def children(self):
        return []
