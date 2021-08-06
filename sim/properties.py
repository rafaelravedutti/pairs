from functools import reduce
from ir.data_types import Type_Float, Type_Vector
from ir.loops import ParticleFor
from ir.memory import Malloc, Realloc
from ir.properties import RegisterProperty, UpdateProperty
from ir.utils import Print
import operator


class PropertiesAlloc:
    def __init__(self, sim, realloc=False):
        self.sim = sim
        self.realloc = realloc

    def lower(self):
        capacity = sum(self.sim.properties.capacities)

        self.sim.clear_block()
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

        return self.sim.block


class PropertiesResetVolatile:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        self.sim.clear_block()
        self.sim.add_statement(Print(self.sim, "PropertiesResetVolatile"))
        for i in ParticleFor(self.sim):
            for p in self.sim.properties.volatiles():
                p[i].set(0.0)

        return self.sim.block
