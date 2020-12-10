from ast.block import Block
from ast.expr import Expr
from ast.branches import Branch
from ast.loops import For


class Timestep:
    def __init__(self, sim, nsteps, item_list=None):
        self.sim = sim
        self.block = Block(sim, [])
        self.timestep_loop = For(sim, 0, nsteps, self.block)

        if item_list is not None:
            for item in item_list:
                if isinstance(item, tuple):
                    self.add(item[0], item[1])
                else:
                    self.add(item)

    def timestep(self):
        return self.timestep_loop.iter()

    def add(self, item, exec_every=0):
        assert exec_every >= 0, \
            "exec_every parameter must be higher or equal than zero!"

        stmts = (item if not isinstance(item, Block)
                 else item.statements())

        ts = self.timestep_loop.iter()
        if exec_every > 0:
            self.block.add_statement(
                Branch(self.sim,
                       Expr.cmp(ts % exec_every, 0),
                       True, Block(self.sim, stmts), None))
        else:
            self.block.add_statement(stmts)

    def as_block(self):
        return Block(self.sim, [self.timestep_loop])

    def generate(self):
        self.block.generate()

    def transform(self, fn):
        self.block = self.block.transform(fn)
