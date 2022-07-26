from pairs.ir.block import pairs_inline
from pairs.ir.contexts import Contexts
from pairs.ir.device import ClearArrayFlag
from pairs.ir.memory import Malloc
from pairs.ir.arrays import ArrayDecl, RegisterArray
from pairs.sim.lowerable import FinalLowerable


class ArraysDecl(FinalLowerable):
    def __init__(self, sim):
        super().__init__(sim)

    @pairs_inline
    def lower(self):
        for a in self.sim.arrays.all():
            alloc_size = a.alloc_size()
            if a.is_static():
                ArrayDecl(self.sim, a)
            else:
                Malloc(self.sim, a, alloc_size, True)

            RegisterArray(self.sim, a, alloc_size)

            if not a.sync():
                ClearArrayFlag(self.sim, self.sim.resizes, Contexts.Host)
                ClearArrayFlag(self.sim, self.sim.resizes, Contexts.Device)
