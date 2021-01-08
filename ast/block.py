from ast.ast_node import ASTNode
from ast.visitor import Visitor


class Block(ASTNode):
    def __init__(self, sim, stmts):
        super().__init__(sim)
        self.level = 0
        self.expressions = []

        if isinstance(stmts, Block):
            self.stmts = stmts.statements()
        else:
            self.stmts = [stmts] if not isinstance(stmts, list) else stmts

    def __lt__(self, other):
        return self.level < other.level

    def __le__(self, other):
        return self.level <= other.level

    def __gt__(self, other):
        return self.level > other.level

    def __ge__(self, other):
        return self.level >= other.level

    def set_level(self, level):
        self.level = level

    def add_statement(self, stmt):
        if isinstance(stmt, list):
            for s in stmt:
                s.parent_block = self

            self.stmts = self.stmts + stmt

        else:
            stmt.parent_block = self
            self.stmts.append(stmt)

    def statements(self):
        return self.stmts

    def add_expression(self, expr):
        if isinstance(expr, list):
            self.expressions = self.expressions + expr
        else:
            self.expressions.append(expr)

    def children(self):
        return self.stmts

    def transform(self, fn):
        for i in range(0, len(self.stmts)):
            self.stmts[i] = self.stmts[i].transform(fn)

        return fn(self)

    def merge_blocks(block1, block2):
        assert isinstance(block1, Block), "First block type is not Block!"
        assert isinstance(block2, Block), "Second block type is not Block!"
        return Block(block1.sim, block1.statements() + block2.statements())

    def from_list(sim, block_list):
        assert isinstance(block_list, list), "Passed argument is not a list!"
        result_block = Block(sim, [])

        for block in block_list:
            assert isinstance(block, Block), "Element in list is not Block!"
            result_block = Block.merge_blocks(result_block, block)

        return result_block
