from ast.block import BlockAST
from ast.branches import FilterAST
from ast.data_types import Type_Int
from ast.expr import ExprAST
from ast.lit import as_lit_ast
from ast.scope import Scope


class IterAST():
    last_iter = 0

    def new_id():
        IterAST.last_iter += 1
        return IterAST.last_iter - 1

    def __init__(self, sim, loop):
        self.sim = sim
        self.loop = loop
        self.iter_id = IterAST.new_id()

    def type(self):
        return Type_Int

    def scope(self):
        return Scope(self.loop.block)

    def __mul__(self, other):
        from ast.expr import ExprAST
        return ExprAST(self.sim, self, other, '*')

    def __rmul__(self, other):
        from ast.expr import ExprAST
        return ExprAST(self.sim, other, self, '*')

    def __eq__(self, other):
        if isinstance(other, IterAST):
            return self.iter_id == other.iter_id

        return False

    def __req__(self, other):
        return self.__cmp__(other)

    def __mod__(self, other):
        from ast.expr import ExprAST
        return ExprAST(self.sim, self, other, '%')

    def __str__(self):
        return f"Iter<{self.iter_id}>"

    def generate(self, mem=False):
        assert mem is False, "Iterator is not lvalue!"
        return f"i{self.iter_id}"

    def transform(self, fn):
        return fn(self)


class ForAST():
    def __init__(self, sim, range_min, range_max, block=None):
        self.sim = sim
        self.iterator = IterAST(sim, self)
        self.min = as_lit_ast(sim, range_min)
        self.max = as_lit_ast(sim, range_max)
        self.parent_block = None
        self.block = BlockAST(sim, []) if block is None else block

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

    def generate(self):
        it_id = self.iterator.generate()
        rmin = self.min.generate()
        rmax = self.max.generate()
        self.sim.code_gen.generate_for_preamble(it_id, rmin, rmax)
        self.block.generate()
        self.sim.code_gen.generate_for_epilogue()

    def transform(self, fn):
        self.iterator = self.iterator.transform(fn)
        self.block = self.block.transform(fn)
        return fn(self)


class ParticleForAST(ForAST):
    def __init__(self, sim, block=None):
        super().__init__(sim, 0, 0, block)

    def __str__(self):
        return f"ParticleFor<>"

    def generate(self):
        self.sim.code_gen.generate_for_preamble(
            self.iterator.generate(), 0, self.sim.nparticles.generate())
        self.block.generate()
        self.sim.code_gen.generate_for_epilogue()


class WhileAST():
    def __init__(self, sim, cond, block=None):
        self.sim = sim
        self.parent_block = None
        self.cond = cond
        self.block = BlockAST(sim, []) if block is None else block

    def __str__(self):
        return f"While<{self.cond}>"

    def __iter__(self):
        self.sim.add_statement(self)
        self.sim.enter_scope(self)
        yield
        self.sim.leave_scope()

    def add_statement(self, stmt):
        self.block.add_statement(stmt)

    def generate(self):
        from ast.expr import ExprAST
        cond_gen = (self.cond.generate() if not isinstance(self.cond, ExprAST)
                    else self.cond.generate_inline())
        self.sim.code_gen.generate_while_preamble(cond_gen)
        self.block.generate()
        self.sim.code_gen.generate_while_epilogue()

    def transform(self, fn):
        self.cond = self.cond.transform(fn)
        self.block = self.block.transform(fn)
        return fn(self)


class NeighborForAST():
    def __init__(self, sim, particle, cell_lists):
        self.sim = sim
        self.parent_block = None
        self.particle = particle
        self.cell_lists = cell_lists

    def __str__(self):
        return f"NeighborFor<particle: {self.particle}>"

    def __iter__(self):
        cl = self.cell_lists
        for s in ForAST(self.sim, 0, cl.nstencil):
            neigh_cell = cl.particle_cell[self.particle] + cl.stencil[s]
            for nc in ForAST(self.sim, 0, cl.cell_sizes[neigh_cell]):
                it = cl.cell_particles[neigh_cell][nc]
                for _ in FilterAST(self.sim, ExprAST.neq(it, self.particle)):
                    yield it
