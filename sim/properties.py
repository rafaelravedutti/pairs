from ast.data_types import Type_Float, Type_Vector
from ast.loops import ParticleFor
from ast.memory import Malloc, Realloc
from ast.utils import Print


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
                if p.flattened:
                    sizes = [capacity * self.sim.dimensions]
                else:
                    sizes = [capacity, self.sim.dimensions]
            else:
                raise Exception("Invalid property type!")

            if self.realloc:
                Realloc(self.sim, p, p.type(), sizes)
            else:
                Malloc(self.sim, p, p.type(), sizes, True)

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
