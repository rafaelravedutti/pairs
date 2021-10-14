from pairs.ir.block import pairs_block
from pairs.ir.memory import Malloc
from pairs.ir.arrays import ArrayDecl


class ArraysDecl:
    def __init__(self, sim):
        self.sim = sim

    @pairs_block
    def lower(self):
        for a in self.sim.arrays.all():
            if a.is_static():
                ArrayDecl(self.sim, a)
            else:
                Malloc(self.sim, a, a.alloc_size(), True)
