from ast.block import BlockAST
from ast.lit import is_literal, LitAST


class BranchAST:
    def __init__(self, sim, cond, block_if, block_else):
        self.sim = sim
        self.cond = LitAST(cond) if is_literal(cond) else cond
        self.block_if = block_if
        self.block_else = block_else

    def if_stmt(sim, cond, body):
        return BranchAST(sim, cond, (body if isinstance(body, BlockAST)
                                     else BlockAST(sim, body)), None)

    def if_else_stmt(sim, cond, body_if, body_else):
        return BranchAST(
            sim, cond,
            (body_if if isinstance(body_if, BlockAST)
             else BlockAST(sim, body_if)),
            (body_else if isinstance(body_else, BlockAST)
             else BlockAST(sim, body_else)))

    def generate(self):
        self.sim.code_gen.generate_if(self.cond.generate())
        self.block_if.generate()

        if self.block_else is not None:
            self.sim.code_gen.generate_else()
            self.block_else.generate()

        self.sim.code_gen.generate_endif()

    def transform(self, fn):
        self.cond = self.cond.transform(fn)
        self.block_if = self.block_if.transform(fn)
        self.block_else = (None if self.block_else is None
                           else self.block_else.transform(fn))
        return fn(self)
