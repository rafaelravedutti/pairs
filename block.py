from printer import printer

class BlockAST:
    def __init__(self, stmts):
        self.stmts = stmts

    def generate(self):
        printer.add_ind(4)
        for stmt in self.stmts:
            stmt.generate()
        printer.add_ind(-4)

    def transform(self, fn):
        for i in range(0, len(self.stmts)):
            self.stmts[i] = self.stmts[i].transform(fn)

        return fn(self)
