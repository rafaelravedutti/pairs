from data_types import Type_Int
from printer import printer

class IterAST():
    def __init__(self, sim):
        self.sim = sim
        self.iter_id = sim.new_iter()

    def type(self):
        return Type_Int

    def __mul__(self, other):
        from expr import ExprAST
        return ExprAST(self.sim, self, other, '*')

    def __rmul__(self, other):
        from expr import ExprAST
        return ExprAST(self.sim, other, self, '*')

    def __str__(self):
        return f"Iter<{self.iter_id}>"

    def generate(self, mem=False):
        assert mem is False, "Iterator is not lvalue!"
        return f"i{self.iter_id}"

    def transform(self, fn):
        return fn(self)

class ForAST():
    def __init__(self, sim, range_min, range_max, body=None):
        self.iterator = IterAST(sim)
        self.min = range_min;
        self.max = range_max;
        self.body = body;

    def iter(self):
        return self.iterator

    def set_body(self, body):
        self.body = body

    def generate(self):
        it_id = self.iterator.generate()
        printer.print(f"for(int {it_id} = {self.min}; {it_id} < {self.max}; {it_id}++) {{")
        self.body.generate();
        printer.print("}")

    def transform(self, fn):
        self.iterator = self.iterator.transform(fn)
        self.body = self.body.transform(fn)
        return fn(self)

class ParticleForAST(ForAST):
    def __init__(self, sim, body=None):
        super().__init__(sim, 0, 0, body)

    def generate(self):
        it_id = self.iterator.generate()
        printer.print(f"for(int {it_id} = 0; {it_id} < nparticles; {it_id}++) {{")
        self.body.generate();
        printer.print("}")

class NeighborForAST(ForAST):
    def __init__(self, sim, particle_iter, body=None):
        super().__init__(sim, 0, 0, body)
        self.particle_iter = particle_iter

    def generate(self):
        it_id = self.iterator.generate()
        printer.print(f"for(int {it_id} = 0; {it_id} < neighbors[{self.particle_iter.generate()}]; {it_id}++) {{")
        self.body.generate();
        printer.print("}")

