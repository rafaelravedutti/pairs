from pairs.ir.block import Block
from pairs.ir.mutator import Mutator


class MergeAdjacentBlocks(Mutator):
    def __init__(self, ast):
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
