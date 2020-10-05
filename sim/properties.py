from ast.data_types import Type_Float, Type_Vector
from ast.loops import ParticleForAST
from ast.memory import MallocAST


class PropertiesDecl:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        nparticles = self.sim.nparticles

        self.sim.clear_block()
        for p in self.sim.properties.all():
            sizes = []
            if p.type() == Type_Float:
                sizes = [nparticles]
            elif p.type() == Type_Vector:
                if p.flattened:
                    sizes = [nparticles * self.sim.dimensions]
                else:
                    sizes = [nparticles, self.sim.dimensions]
            else:
                raise Exception("Invalid property type!")

            MallocAST(self.sim, p, p.type(), sizes, True)

        return self.sim.block


class PropertiesResetVolatile:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        self.sim.clear_block()
        for i in ParticleForAST(self.sim):
            for p in self.sim.properties.volatiles():
                p[i].set(0.0)

        return self.sim.block
