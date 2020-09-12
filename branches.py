from block import BlockAST
from lit import is_literal, LitAST
from printer import printer

class BranchAST:
    def __init__(self, cond, block_if, block_else):
        self.cond = LitAST(cond) if is_literal(cond) else cond
        self.block_if = block_if
        self.block_else = block_else

    def if_stmt(cond, body):
        return BranchAST(cond, body if isinstance(body, BlockAST) else BlockAST(body), None)

    def if_else_stmt(cond, body_if, body_else):
        return BranchAST(cond,
            body_if if isinstance(body_if, BlockAST) else BlockAST(body_if),
            body_else if isinstance(body_else, BlockAST) else BlockAST(body_else)
        )

    def generate(self):
        cvname = self.cond.generate()
        printer.print(f"if({cvname}) {{")
        self.block_if.generate()

        if self.block_else is not None:
            printer.print("} else {")
            self.block_else.generate()

        printer.print("}")

    def transform(self, fn):
        self.cond = self.cond.transform(fn)
        self.block_if = self.block_if.transform(fn)
        self.block_else = None if self.block_else is None else self.block_else.transform(fn)
        return fn(self)
