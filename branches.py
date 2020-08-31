from printer import printer

class BranchAST:
    def __init__(self, cond, block_if, block_else):
        self.cond = cond
        self.block_if = block_if
        self.block_else = block_else

    def generate(self):
        cvname = self.cond.generate()
        printer.print("if({}) {{".format(cvname))

        printer.add_ind(4)
        for stmt in self.block_if:
            stmt.generate()
        printer.add_ind(-4)

        if self.block_else is not None:
            printer.print("} else {")
            printer.add_ind(4)
            for stmt in self.block_else:
                stmt.generate()
            printer.add_ind(-4)

        printer.print("}")
