from pairs.ir.ast_node import ASTNode
from pairs.ir.properties import Property
from pairs.ir.features import Feature


class ParticleAttributeList(ASTNode):
    def __init__(self, sim, items):
        super().__init__(sim)
        self.list = []

        for i in items:
            if isinstance(i, Property):
                self.list.append(i)

            if isinstance(i, Feature):
                self.list.append(i.prop())

            if isinstance(i, str):
                prop = sim.property(i)
                if prop is None:
                    prop = sim.feature(i).prop()

                assert prop is not None, f"Attribute not found: {i}"
                self.list.append(prop)

    def __iter__(self):
        yield from self.list

    def length(self):
        return len(self.list)
