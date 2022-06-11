from pairs.ir.mutator import Mutator
from pairs.sim.lowerable import Lowerable, FinalLowerable


class Lower(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.lowered_nodes = 0
        self.lower_finals = False

    def set_data(self, data):
        self.lower_finals = data[0]

    def mutate_Unknown(self, ast_node):
        if isinstance(ast_node, Lowerable) or (self.lower_finals and isinstance(ast_node, FinalLowerable)):
            self.lowered_nodes += 1
            return ast_node.lower()

        return ast_node
