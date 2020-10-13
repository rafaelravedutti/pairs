from ast.block import Block
from ast.lit import as_lit_ast


class Branch:
    def __init__(self, sim, cond, one_way=False, blk_if=None, blk_else=None):
        self.sim = sim
        self.parent_block = None
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

    def generate(self):
        cond_gen = self.cond.generate()
        self.sim.code_gen.generate_if(cond_gen)
        self.block_if.generate()

        if self.block_else is not None:
            self.sim.code_gen.generate_else()
            self.block_else.generate()

        self.sim.code_gen.generate_endif()

    def transform(self, fn):
        self.cond = self.cond.transform(fn)
        self.block_if = self.block_if.transform(fn)
        self.block_else = \
            None if self.block_else is None \
            else self.block_else.transform(fn)
        return fn(self)


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
