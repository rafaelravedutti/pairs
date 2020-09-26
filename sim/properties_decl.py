from ast.block import BlockAST
from ast.properties import PropertyDeclAST

class PropertiesDecl:
    def __init__(self, sim):
        self.sim = sim

    def lower(self):
        decls = []
        for p in self.sim.properties.all():
            decls.append(PropertyDeclAST(self.sim, p, self.sim.nparticles))

        return BlockAST(decls)
