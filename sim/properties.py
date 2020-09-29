from ast.block import BlockAST
from ast.properties import PropertyDeclAST
from ast.loops import ParticleForAST


class PropertiesDecl:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        decls = []
        for p in self.sim.properties.all():
            decls.append(PropertyDeclAST(self.sim, p, self.sim.nparticles))

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
