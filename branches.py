from printer import printer

class BranchAST:
    def __init__(self, cond, block_if, block_else):
        self.cond = cond
        self.block_if = block_if
        self.block_else = block_else

    def generate(self):
        cvname = self.cond.generate()
        printer.print(f"if({cvname}) {{")
        printer.add_ind(4)
        self.block_if.generate()
        printer.add_ind(-4)

        if self.block_else is not None:
            printer.print("} else {")
            printer.add_ind(4)
            self.block_else.generate()
            printer.add_ind(-4)

        printer.print("}")

    def transform(self, fn):
        self.cond = self.cond.transform(fn)
        self.block_if = self.block_if.transform(fn)
        self.block_else = None if self.block_else is None else self.block_else.transform(fn)
        return fn(self)
