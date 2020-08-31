from printer import printer

class IterAST():
    def __init__(self, sim):
        self.iter_id = sim.new_iter()

    def generate(self):
        return f"i{self.iter_id}"

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
        printer.print(f"for(int {it_id} = {self.min}; {it_id} < {self.max}; i++) {{")
        printer.add_ind(4)
        self.body.generate();
        printer.add_ind(-4)
        printer.print("}")

class ParticleForAST(ForAST):
    def __init__(self, sim, body=None):
        super().__init__(sim, 0, 0, body)

    def generate(self):
        it_id = self.iterator.generate()
        printer.print(f"for(int {it_id} = 0; {it_id} < nparticles; {it_id}++) {{")
        printer.add_ind(4)
        self.body.generate();
        printer.add_ind(-4)
        printer.print("}")

class NeighborForAST(ForAST):
    def __init__(self, sim, particle_iter, body=None):
        super().__init__(sim, 0, 0, body)
        self.particle_iter = particle_iter

    def generate(self):
        it_id = self.iterator.generate()
        printer.print(f"for(int {it_id} = 0; {it_id} < neighbors[{self.particle_iter.generate()}]; {it_id}++) {{")
        printer.add_ind(4)
        self.body.generate();
        printer.add_ind(-4)
        printer.print("}")

