from pairs.ir.ast_node import ASTNode


class Lowerable:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        raise Exception("Error: lower() method must be implemented for Lowerable inherited classes!")


class FinalLowerable(ASTNode):
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        raise Exception("Error: lower() method must be implemented for FinalLowerable inherited classes!")
