from pairs.ir.scalars import ScalarOp
from pairs.ir.block import Block
from pairs.ir.branches import Branch, Filter
from pairs.ir.loops import For


class Timestep:
    def __init__(self, sim, nsteps, item_list=None):
        self.sim = sim
        self.block = Block(sim, [])
        self.timestep_loop = For(sim, 0, nsteps + 1, self.block)

        if item_list is not None:
            for item in item_list:
                if isinstance(item, tuple):
                    stmt_else = None

                    if len(item) == 2:
                        stmt, params = item

                    if len(item) == 3:
                        stmt, stmt_else, params = item

                    exec_every = 0 if 'every' not in params else params['every']
                    skip_first = False if 'skip_first' not in params else params['skip_first']
                    self.add(stmt, exec_every, stmt_else, skip_first)

                else:
                    self.add(item)

    def timestep(self):
        return self.timestep_loop.iter()

    def add(self, item, exec_every=0, item_else=None, skip_first=False):
        assert exec_every >= 0, "exec_every parameter must be higher or equal than zero!"
        stmts = item if not isinstance(item, Block) else item.statements()
        stmts_else = None
        ts = self.timestep_loop.iter()
        self.sim.enter(self.block)

        if item_else is not None:
            stmts_else = item_else if not isinstance(item_else, Block) else item_else.statements()

        if exec_every > 0:
            cond = ScalarOp.or_op(ScalarOp.cmp((ts + 1) % exec_every, 0), ScalarOp.cmp(ts, 0))
            one_way = True if stmts_else is None else False

            self.block.add_statement(
                Branch(self.sim, ScalarOp.inline(cond), one_way,
                    Block(self.sim, stmts),
                    Block(self.sim, stmts_else) if not one_way else None))

        elif skip_first:
            self.block.add_statement(Filter(self.sim, ScalarOp.inline(ts > 0), Block(self.sim, stmts)))

        else:
            self.block.add_statement(stmts)

        self.sim.leave()

    def as_block(self):
        return Block(self.sim, [self.timestep_loop])
