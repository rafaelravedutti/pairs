from ast.data_types import Type_Int

class SizeofAST:
    def __init__(self, sim, data_type):
        self.sim = sim
        self.data_type = data_type

    def type(self):
        return Type_Int

    def generate(self, mem=False):
        return self.sim.code_gen.generate_sizeof(self.data_type)

    def transform(self, fn):
        return fn(self)