from ast.data_types import Type_Invalid


class ASTNode:
    def __init__(self, sim):
        self.sim = sim
        self._parent_block = None # Set during SetParentBlock transformation

    def __str__(self):
        return "ASTNode<>"

    def type(self):
        return Type_Invalid

    def scope(self):
        return self.sim.global_scope

    @property
    def parent_block(self):
        return self._parent_block

    def children(self):
        return []
