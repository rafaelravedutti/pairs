from ast.visitor import Visitor

class BlockAST:
    def __init__(self, sim, stmts):
        self.sim = sim
        self.level = 0
        self.expressions = []

        if isinstance(stmts, BlockAST):
            self.stmts = stmts.statements()
        else:
            self.stmts = [stmts] if not isinstance(stmts, list) else stmts

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

    def generate(self):
        self.sim.code_gen.generate_block_preamble()

        for expr in self.expressions:
            expr.generate()

        for stmt in self.stmts:
            stmt.generate()

        self.sim.code_gen.generate_block_epilogue()

    def transform(self, fn):
        for i in range(0, len(self.stmts)):
            self.stmts[i] = self.stmts[i].transform(fn)

        return fn(self)

    def merge_blocks(block1, block2):
        assert isinstance(block1, BlockAST), \
            "First block type is not BlockAST!"
        assert isinstance(block2, BlockAST), \
            "Second block type is not BlockAST!"
        return BlockAST(block1.sim, block1.statements() + block2.statements())

    def from_list(sim, block_list):
        assert isinstance(block_list, list), "Passed argument is not a list!"
        result_block = BlockAST(sim, [])

        for block in block_list:
            assert isinstance(block, BlockAST), \
                "Element in list is not BlockAST!"
            result_block = BlockAST.merge_blocks(result_block, block)

        return result_block

    def set_block_levels(ast):
        BlockAST.level = 0

        def enter(ast):
            if isinstance(ast, BlockAST):
                ast.set_level(BlockAST.level)
                BlockAST.level += 1

        def leave(ast):
            if isinstance(ast, BlockAST):
                BlockAST.level -= 1

        Visitor(ast, enter, leave).visit()
