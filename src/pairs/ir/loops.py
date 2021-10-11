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

    def scope(self):
        return self.loop.block

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
        self.sim.enter_scope(self)
        yield self.iterator
        self.sim.leave_scope()

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
        self.sim.enter_scope(self)
        yield
        self.sim.leave_scope()

    def add_statement(self, stmt):
        self.block.add_statement(stmt)

    def children(self):
        return [self.cond, self.block]


class NeighborFor():
    def __init__(self, sim, particle, cell_lists, neighbor_lists=None):
        self.sim = sim
        self.particle = particle
        self.cell_lists = cell_lists
        self.neighbor_lists = neighbor_lists

    def __str__(self):
        return f"NeighborFor<particle: {self.particle}>"

    def __iter__(self):
        if self.neighbor_lists is None:
            cl = self.cell_lists
            for s in For(self.sim, 0, cl.nstencil):
                neigh_cell = cl.particle_cell[self.particle] + cl.stencil[s]
                for _ in Filter(self.sim, BinOp.and_op(neigh_cell >= 0, neigh_cell <= cl.ncells)):
                    for nc in For(self.sim, 0, cl.cell_sizes[neigh_cell]):
                        it = cl.cell_particles[neigh_cell][nc]
                        for _ in Filter(self.sim, BinOp.neq(it, self.particle)):
                                yield it
        else:
            neighbor_lists = self.neighbor_lists
            for k in For(self.sim, 0, neighbor_lists.numneighs[self.particle]):
                yield neighbor_lists.neighborlists[self.particle][k]
