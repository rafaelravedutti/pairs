from ast.data_types import Type_Float, Type_Vector
from ast.loops import ParticleFor
from ast.memory import Malloc
from ast.utils import Print


class PropertiesDecl:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        particle_capacity = self.sim.particle_capacity

        self.sim.clear_block()
        for p in self.sim.properties.all():
            sizes = []
            if p.type() == Type_Float:
                sizes = [particle_capacity]
            elif p.type() == Type_Vector:
                if p.flattened:
                    sizes = [particle_capacity * self.sim.dimensions]
                else:
                    sizes = [particle_capacity, self.sim.dimensions]
            else:
                raise Exception("Invalid property type!")

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
