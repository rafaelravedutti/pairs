from ast.ast_node import ASTNode


class Block(ASTNode):
    def __init__(self, sim, stmts):
        super().__init__(sim)
        self.level = 0
        self.variants = set()

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
            self.stmts = self.stmts + stmt
        else:
            self.stmts.append(stmt)

    def add_variant(self, variant):
        for v in variant if isinstance(variant, list) else [variant]:
            self.variants.add(v)

    def statements(self):
        return self.stmts

    def children(self):
        return self.stmts

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
