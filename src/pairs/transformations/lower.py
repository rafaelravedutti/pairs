from pairs.ir.mutator import Mutator
from pairs.sim.lowerable import Lowerable


class Lower(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        self.lowered_nodes = 0

    def mutate_Unknown(self, ast_node):
        if isinstance(ast_node, Lowerable):
            self.lowered_nodes += 1
            return ast_node.lower()

        return ast_node


def lower_everything(ast):
    nlowered = 1
    while nlowered > 0:
        lower = Lower(ast)
        lower.mutate()
        nlowered = lower.lowered_nodes
