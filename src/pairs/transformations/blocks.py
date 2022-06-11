from pairs.ir.block import Block
from pairs.ir.bin_op import Decl
from pairs.ir.mutator import Mutator


class MergeAdjacentBlocks(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)

    def mutate_Block(self, ast_node):
        new_stmts = []
        stmts = [self.mutate(s) for s in ast_node.stmts]

        for s in stmts:
            if isinstance(s, Block):
                new_stmts = new_stmts + s.statements()
            else:
                new_stmts.append(s)

        ast_node.stmts = new_stmts 
        return ast_node


class LiftExprOwnerBlocks(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.ownership = None
        self.expressions_to_lift = None

    def set_data(self, data):
        self.ownership = data[0]
        self.expressions_to_lift = data[1]

    def mutate_Block(self, ast_node):
        ast_node.stmts = \
            [Decl(ast_node.sim, e) for e in self.ownership if self.ownership[e] == ast_node and e in self.expressions_to_lift] + \
            [self.mutate(s) for s in ast_node.stmts]
        return ast_node
