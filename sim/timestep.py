from ast.block import BlockAST
from ast.expr import ExprAST
from ast.branches import BranchAST
from ast.loops import ForAST


class Timestep:
    def __init__(self, sim, nsteps, item_list=None):
        self.sim = sim
        self.block = BlockAST(sim, [])
        self.timestep_loop = ForAST(sim, 0, nsteps, self.block)

        if item_list is not None:
            for item in item_list:
                if isinstance(item, tuple):
                    self.add(item[0], item[1])
                else:
                    self.add(item)

    def add(self, item, exec_every=0):
        assert exec_every >= 0, \
            "exec_every parameter must be higher or equal than zero!"

        statements = (item if not isinstance(item, BlockAST)
                      else item.statements())

        ts = self.timestep_loop.iter()

        if exec_every > 0:
            self.block.add_statement(
                BranchAST.if_stmt(
                    self.sim, ExprAST.cmp(ts % exec_every, 0), statements))
        else:
            self.block.add_statement(statements)

    def as_block(self):
        return BlockAST(self.sim, [self.timestep_loop])

    def generate(self):
        self.block.generate()

    def transform(self, fn):
        self.block = self.block.transform(fn)
