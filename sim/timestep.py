from ast.block import BlockAST
from ast.expr import ExprAST
from ast.branches import BranchAST
from ast.loops import ForAST

class Timestep:
    def __init__(self, sim, nsteps):
        self.sim = sim
        self.block = BlockAST([])
        self.timestep_loop = ForAST(sim, 0, nsteps, self.block)

    def add(self, item, exec_every=0):
        assert exec_every >= 0, "Timestep frequency parameter must be higher or equal than zero!"
        statements = item if not isinstance(item, BlockAST) else item.statements()
        if exec_every > 0:
            self.block.add_statement(BranchAST.if_stmt(ExprAST.cmp(self.timestep_loop.iter() % exec_every, 0), statements))
        else:
            self.block.add_statement(statements)

    def as_block(self):
        return BlockAST([self.timestep_loop])

    def generate(self):
        self.block.generate()

    def transform(self, fn):
        self.block = self.block.transform(fn)
