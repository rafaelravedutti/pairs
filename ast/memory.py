class ReallocAST:
    def __init__(self, sim, array, size):
        self.sim = sim
        self.array = array
        self.size = size

    def generate(self, mem=False):
        self.sim.code_gen.generate_realloc(self.array.generate(), self.size.generate())

    def transform(self, fn):
        self.array = self.array.transform(fn)
        self.size = self.size.transform(fn)
        return fn(self)
