from ast.block import BlockAST
from ast.data_types import Type_Float, Type_Vector
from ast.loops import ParticleForAST
from ast.memory import MallocAST

class PropertiesDecl:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        nparticles = self.sim.nparticles
        decls = []

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

            decls.append(MallocAST(self.sim, p, p.type(), sizes, True))

        return BlockAST(self.sim, decls)


class PropertiesResetVolatile:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        decls = []
        reset_loop = ParticleForAST(self.sim)

        for p in self.sim.properties.volatiles():
            decls.append(p[reset_loop.iter()].set(0.0))

        reset_loop.set_body(BlockAST(self.sim, decls))
        return reset_loop
