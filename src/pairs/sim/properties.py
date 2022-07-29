from pairs.ir.block import pairs_device_block, pairs_inline
from pairs.ir.loops import ParticleFor
from pairs.ir.memory import Malloc, Realloc
from pairs.ir.properties import RegisterProperty
from pairs.ir.types import Types
from pairs.ir.utils import Print
from pairs.sim.lowerable import Lowerable, FinalLowerable
from functools import reduce
import operator


class PropertiesAlloc(FinalLowerable):
    def __init__(self, sim, realloc=False):
        self.sim = sim
        self.realloc = realloc

    @pairs_inline
    def lower(self):
        capacity = sum(self.sim.properties.capacities)
        for p in self.sim.properties.all():
            sizes = []
            if Types.is_real(p.type()):
                sizes = [capacity]
            elif p.type() == Types.Vector:
                sizes = [capacity, self.sim.ndims()]
            else:
                raise Exception("Invalid property type!")

            if self.realloc:
                UpdateProperty(self.sim, p, sizes)
            else:
                RegisterProperty(self.sim, p, sizes)


class PropertiesResetVolatile(Lowerable):
    def __init__(self, sim):
        self.sim = sim

    @pairs_device_block
    def lower(self):
        self.sim.module_name("reset_volatile_properties")
        for i in ParticleFor(self.sim):
            for p in self.sim.properties.volatiles():
                p[i].set(0.0)
