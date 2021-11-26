from pairs.ir.block import pairs_block, pairs_device_block
from pairs.ir.data_types import Type_Float, Type_Vector
from pairs.ir.loops import ParticleFor
from pairs.ir.memory import Malloc, Realloc
from pairs.ir.properties import RegisterProperty, UpdateProperty
from pairs.ir.utils import Print
from pairs.sim.lowerable import Lowerable
from functools import reduce
import operator


class PropertiesAlloc(Lowerable):
    def __init__(self, sim, realloc=False):
        self.sim = sim
        self.realloc = realloc

    @pairs_block
    def lower(self):
        capacity = sum(self.sim.properties.capacities)
        for p in self.sim.properties.all():
            sizes = []
            if p.type() == Type_Float:
                sizes = [capacity]
            elif p.type() == Type_Vector:
                sizes = [capacity, self.sim.ndims()]
            else:
                raise Exception("Invalid property type!")

            if self.realloc:
                Realloc(self.sim, p, reduce(operator.mul, sizes))
                UpdateProperty(self.sim, p, sizes)
            else:
                Malloc(self.sim, p, reduce(operator.mul, sizes), True)
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
