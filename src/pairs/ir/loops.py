from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.lit import Lit
from pairs.ir.types import Types


class Iter(ASTTerm):
    last_iter = 0

    def new_id():
        Iter.last_iter += 1
        return Iter.last_iter - 1

    def __init__(self, sim, loop):
        super().__init__(sim, ScalarOp)
        self.loop = loop
        self.iter_id = Iter.new_id()

    def id(self):
        return self.iter_id

    def name(self):
        return f"i{self.iter_id}"

    def type(self):
        return Types.Int32

    def __eq__(self, other):
        return isinstance(other, Iter) and self.iter_id == other.iter_id

    def __req__(self, other):
        return self.__cmp__(other)

    def __str__(self):
        return f"Iter<{self.iter_id}>"


class For(ASTNode):
    def __init__(self, sim, range_min, range_max, block=None):
        super().__init__(sim)
        self.iterator = Iter(sim, self)
        self.min = Lit.cvt(sim, range_min)
        self.max = Lit.cvt(sim, range_max)
        self.block = Block(sim, []) if block is None else block
        self.kernel = None
        self._kernel_candidate = False

    def __str__(self):
        return f"For<{self.iterator}, {self.min} ... {self.max}>"

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter(self)
        yield self.iterator
        self.sim.leave()

    def iter(self):
        return self.iterator

    def mark_as_kernel_candidate(self):
        self._kernel_candidate = True

    def is_kernel_candidate(self):
        return self._kernel_candidate

    def add_statement(self, stmt):
        self.block.add_statement(stmt)

    def children(self):
        return [self.iterator, self.block, self.min, self.max]


class ParticleFor(For):
    def __init__(self, sim, block=None, local_only=True):
        super().__init__(sim, 0, sim.nlocal if local_only else sim.nlocal + sim.nghost, block)
        self.local_only = local_only

    def __str__(self):
        return f"ParticleFor<self.iterator>"

    def children(self):
        return [self.block, self.sim.nlocal] + ([] if self.local_only else [self.sim.nghost])


class While(ASTNode):
    def __init__(self, sim, cond, block=None):
        super().__init__(sim)
        self.cond = ScalarOp.inline(cond)
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

    def is_kernel_candidate(self):
        return False

    def children(self):
        return [self.cond, self.block]


class Continue(ASTNode):
    def __init__(self, sim):
        super().__init__(sim)

    def __str__(self):
        return f"Continue<>"

    def __call__(self):
        self.sim.add_statement(self)
