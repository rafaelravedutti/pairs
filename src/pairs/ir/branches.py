from pairs.ir.ast_node import ASTNode
from pairs.ir.block import Block
from pairs.ir.lit import as_lit_ast


class Branch(ASTNode):
    def __init__(self, sim, cond, one_way=False, blk_if=None, blk_else=None):
        self.sim = sim
        self.cond = as_lit_ast(sim, cond)
        self.switch = True
        self.block_if = Block(sim, []) if blk_if is None else blk_if
        self.block_else = \
            None if one_way \
            else Block(sim, []) if blk_else is None \
            else blk_else

    def __iter__(self):
        self.sim.add_statement(self)
        self.switch = True
        self.sim.enter_scope(self)
        yield self.switch
        self.sim.leave_scope()

        self.switch = False
        self.sim.enter_scope(self)
        yield self.switch
        self.sim.leave_scope()

    def add_statement(self, stmt):
        if self.switch:
            self.block_if.add_statement(stmt)
        else:
            self.block_else.add_statement(stmt)

    def children(self):
        return [self.cond, self.block_if] + \
               ([] if self.block_else is None else [self.block_else])


class Filter(Branch):
    def __init__(self, sim, cond):
        super().__init__(sim, cond, True)

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter_scope(self)
        yield
        self.sim.leave_scope()

    def add_statement(self, stmt):
        self.block_if.add_statement(stmt)
