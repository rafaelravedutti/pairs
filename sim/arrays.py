from ast.memory import Malloc


class ArraysDecl:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        self.sim.clear_block()
        for a in self.sim.arrays.all():
            Malloc(self.sim, a, a.type(), a.alloc_size(), True)

        return self.sim.block
