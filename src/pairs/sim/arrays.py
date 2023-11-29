from pairs.ir.block import pairs_inline
from pairs.ir.contexts import Contexts
from pairs.ir.memory import Malloc
from pairs.ir.arrays import DeclareStaticArray, RegisterArray
from pairs.sim.lowerable import FinalLowerable


class DeclareArrays(FinalLowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        for a in self.sim.arrays.all():
            if a.is_static():
                DeclareStaticArray(self.sim, a)

            RegisterArray(self.sim, a, a.alloc_size())
