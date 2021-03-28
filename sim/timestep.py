from ir.bin_op import BinOp
from ir.block import Block
from ir.branches import Branch
from ir.loops import For


class Timestep:
    def __init__(self, sim, nsteps, item_list=None):
        self.sim = sim
        self.block = Block(sim, [])
        self.timestep_loop = For(sim, 0, nsteps + 1, self.block)

        if item_list is not None:
            for item in item_list:
                if isinstance(item, tuple):
                    if len(item) >= 3:
                        self.add(item[0], item[2], item[1])
                    else:
                        self.add(item[0], item[1])
                else:
                    self.add(item)

    def timestep(self):
        return self.timestep_loop.iter()

    def add(self, item, exec_every=0, item_else=None):
        assert exec_every >= 0, "exec_every parameter must be higher or equal than zero!"
        stmts = item if not isinstance(item, Block) else item.statements()
        stmts_else = None
        ts = self.timestep_loop.iter()

        if item_else is not None:
            stmts_else = item_else if not isinstance(item_else, Block) else item_else.statements()

        if exec_every > 0:
            self.block.add_statement(
                Branch(self.sim, BinOp.cmp(ts % exec_every, 0), True if stmts_else is None else False,
                Block(self.sim, stmts), None if stmts_else is None else Block(self.sim, stmts_else)))
        else:
            self.block.add_statement(stmts)

    def as_block(self):
        return Block(self.sim, [self.timestep_loop])
