from printer import printer

class BlockAST:
    def __init__(self, stmts):
        if isinstance(stmts, BlockAST):
            self.stmts = stmts.statements()
        else:
            self.stmts = [stmts] if not isinstance(stmts, list) else stmts

    def add_statement(self, stmt):
        if isinstance(stmt, list):
            self.stmts = self.stmts + stmt
        else:
            self.stmts.append(stmt)

    def statements(self):
        return self.stmts

    def generate(self):
        printer.add_ind(4)
        for stmt in self.stmts:
            stmt.generate()
        printer.add_ind(-4)

    def transform(self, fn):
        for i in range(0, len(self.stmts)):
            self.stmts[i] = self.stmts[i].transform(fn)

        return fn(self)

    def merge_blocks(block1, block2):
        assert isinstance(block1, BlockAST), "First block type is not BlockAST!"
        assert isinstance(block2, BlockAST), "Second block type is not BlockAST!"
        return BlockAST(block1.statements() + block2.statements())
