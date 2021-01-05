from ast.block import Block
from ast.branches import Filter
from ast.data_types import Type_Int
from ast.expr import BinOp
from ast.lit import as_lit_ast


class Iter():
    last_iter = 0

    def new_id():
        Iter.last_iter += 1
        return Iter.last_iter - 1

    def __init__(self, sim, loop):
        self.sim = sim
        self.loop = loop
        self.iter_id = Iter.new_id()

    def id(self):
        return self.iter_id

    def type(self):
        return Type_Int

    def is_mutable(self):
        # FIXME: This should be set to False, but currently the flattening transformation is reusing
        # expressions that are not alive anymore (used first within if, then outside it), causing
        # the generated code to be uncompilable
        return True

    def scope(self):
        return self.loop.block

    def __add__(self, other):
        return BinOp(self.sim, self, other, '+')

    def __sub__(self, other):
        return BinOp(self.sim, self, other, '-')

    def __mul__(self, other):
        from ast.expr import BinOp
        return BinOp(self.sim, self, other, '*')

    def __rmul__(self, other):
        from ast.expr import BinOp
        return BinOp(self.sim, other, self, '*')

    def __eq__(self, other):
        if isinstance(other, Iter):
            return self.iter_id == other.iter_id

        return False

    def __req__(self, other):
        return self.__cmp__(other)

    def __mod__(self, other):
        from ast.expr import BinOp
        return BinOp(self.sim, self, other, '%')

    def __str__(self):
        return f"Iter<{self.iter_id}>"

    def children(self):
        return []

    def transform(self, fn):
        return fn(self)


class For():
    def __init__(self, sim, range_min, range_max, block=None):
        self.sim = sim
        self.iterator = Iter(sim, self)
        self.min = as_lit_ast(sim, range_min)
        self.max = as_lit_ast(sim, range_max)
        self.parent_block = None
        self.block = Block(sim, []) if block is None else block

    def __str__(self):
        return f"For<min: {self.min}, max: {self.max}>"

    def iter(self):
        return self.iterator

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter_scope(self)
        yield self.iterator
        self.sim.leave_scope()

    def add_statement(self, stmt):
        self.block.add_statement(stmt)

    def children(self):
        return [self.iterator, self.block]

    def transform(self, fn):
        self.iterator = self.iterator.transform(fn)
        self.block = self.block.transform(fn)
        return fn(self)


class ParticleFor(For):
    def __init__(self, sim, block=None, local_only=True):
        super().__init__(sim, 0, 0, block)
        self.local_only = local_only

    def __str__(self):
        return f"ParticleFor<>"


class While():
    def __init__(self, sim, cond, block=None):
        from ast.expr import BinOp
        self.sim = sim
        self.parent_block = None
        self.cond = BinOp.inline(cond)
        self.block = Block(sim, []) if block is None else block

    def __str__(self):
        return f"While<{self.cond}>"

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter_scope(self)
        yield
        self.sim.leave_scope()

    def add_statement(self, stmt):
        self.block.add_statement(stmt)

    def children(self):
        return [self.cond, self.block]

    def transform(self, fn):
        self.cond = self.cond.transform(fn)
        self.block = self.block.transform(fn)
        return fn(self)


class NeighborFor():
    def __init__(self, sim, particle, cell_lists):
        self.sim = sim
        self.parent_block = None
        self.particle = particle
        self.cell_lists = cell_lists

    def __str__(self):
        return f"NeighborFor<particle: {self.particle}>"

    def __iter__(self):
        cl = self.cell_lists
        for s in For(self.sim, 0, cl.nstencil):
            neigh_cell = cl.particle_cell[self.particle] + cl.stencil[s]
            for _ in Filter(self.sim,
                            BinOp.and_op(neigh_cell >= 0,
                                         neigh_cell <= cl.ncells_all)):
                for nc in For(self.sim, 0, cl.cell_sizes[neigh_cell]):
                    it = cl.cell_particles[neigh_cell][nc]
                    for _ in Filter(self.sim, BinOp.neq(it, self.particle)):
                            yield it
