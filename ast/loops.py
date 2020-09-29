from ast.data_types import Type_Int
from ast.lit import is_literal, LitAST

class IterAST():
    def __init__(self, sim):
        self.sim = sim
        self.iter_id = sim.new_iter()

    def type(self):
        return Type_Int

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
    def __init__(self, sim, range_min, range_max, body=None):
        self.sim = sim
        self.iterator = IterAST(sim)
        self.min = LitAST(range_min) if is_literal(range_min) else range_min;
        self.max = LitAST(range_max) if is_literal(range_max) else range_max;
        self.body = body;

    def __str__(self):
        return f"For<min: {self.min}, max: {self.max}>"

    def iter(self):
        return self.iterator

    def set_body(self, body):
        self.body = body

    def generate(self):
        it_id = self.iterator.generate()
        rmin = self.min.generate()
        rmax = self.max.generate()
        self.sim.code_gen.generate_for_preamble(it_id, rmin, rmax)
        self.body.generate()
        self.sim.code_gen.generate_for_epilogue()

    def transform(self, fn):
        self.iterator = self.iterator.transform(fn)
        self.body = self.body.transform(fn)
        return fn(self)

class ParticleForAST(ForAST):
    def __init__(self, sim, body=None):
        super().__init__(sim, 0, 0, body)

    def generate(self):
        it_id = self.iterator.generate()
        self.sim.code_gen.generate_for_preamble(it_id, 0, self.sim.nparticles.generate())
        self.body.generate()
        self.sim.code_gen.generate_for_epilogue()

class NeighborForAST(ForAST):
    def __init__(self, sim, particle_iter, body=None):
        super().__init__(sim, 0, 0, body)
        self.particle_iter = particle_iter

    def generate(self):
        it_id = self.iterator.generate()
        self.sim.code_gen.generate_for_preamble(it_id, 0, f"neighbors[{self.particle_iter.generate()}]")
        self.body.generate()
        self.sim.code_gen.generate_for_epilogue()

class WhileAST():
    def __init__(self, sim, cond, body=None):
        self.sim = sim
        self.cond = cond
        self.body = body

    def __str__(self):
        return f"While<{self.cond}>"

    def generate(self):
        from ast.expr import ExprAST
        cond_gen = self.cond.generate() if not isinstance(self.cond, ExprAST) else self.cond.generate_inline()
        self.sim.code_gen.generate_while_preamble(cond_gen)
        self.body.generate()
        self.sim.code_gen.generate_while_epilogue()

    def transform(self, fn):
        self.cond = self.cond.transform(fn)
        self.body = self.body.transform(fn)
        return fn(self)
