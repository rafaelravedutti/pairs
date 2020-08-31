class BlockAST:
    def __init__(self, stmts):
        self.stmts = stmts

    def generate(self):
        for stmt in self.stmts:
            stmt.generate()
