from pairs.ir.ast_node import ASTNode
from pairs.ir.bin_op import BinOp, ASTTerm
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.data_types import Type_Int
from pairs.ir.lit import as_lit_ast


class Iter(ASTTerm):
    last_iter = 0

    def new_id():
        Iter.last_iter += 1
        return Iter.last_iter - 1

    def __init__(self, sim, loop):
        super().__init__(sim)
        self.loop = loop
        self.iter_id = Iter.new_id()

    def id(self):
        return self.iter_id

    def name(self):
        return f"i{self.iter_id}"

    def type(self):
        return Type_Int

    def __eq__(self, other):
        if isinstance(other, Iter):
            return self.iter_id == other.iter_id

        return False

    def __req__(self, other):
        return self.__cmp__(other)

    def __str__(self):
        return f"Iter<{self.iter_id}>"


class For(ASTNode):
    def __init__(self, sim, range_min, range_max, block=None):
        super().__init__(sim)
        self.iterator = Iter(sim, self)
        self.min = as_lit_ast(sim, range_min)
        self.max = as_lit_ast(sim, range_max)
        self.block = Block(sim, []) if block is None else block

    def __str__(self):
        return f"For<min: {self.min}, max: {self.max}>"

    def iter(self):
        return self.iterator

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter(self)
        yield self.iterator
        self.sim.leave()

    def add_statement(self, stmt):
        self.block.add_statement(stmt)

    def children(self):
        return [self.iterator, self.block, self.min, self.max]


class ParticleFor(For):
    def __init__(self, sim, block=None, local_only=True):
        super().__init__(sim, 0, 0, block)
        self.local_only = local_only

    def __str__(self):
        return f"ParticleFor<>"


class While(ASTNode):
    def __init__(self, sim, cond, block=None):
        super().__init__(sim)
        self.cond = BinOp.inline(cond)
        self.block = Block(sim, []) if block is None else block

    def __str__(self):
        return f"While<{self.cond}>"

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter(self)
        yield
        self.sim.leave()

    def add_statement(self, stmt):
        self.block.add_statement(stmt)

    def children(self):
        return [self.cond, self.block]
