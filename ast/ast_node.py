from ast.data_types import Type_Invalid


class ASTNode:
    def __init__(self, sim):
        self.sim = sim

    def __str__(self):
        return "ASTNode<>"

    def type(self):
        return Type_Invalid

    def scope(self):
        return self.sim.global_scope

    def children(self):
        return []

    def transform(self, fn):
        return fn(self)
