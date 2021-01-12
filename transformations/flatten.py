from ast.layouts import Layout_AoS, Layout_SoA
from ast.mutator import Mutator


class FlattenPropertyAccesses(Mutator):
    def __init__(self, ast):
        super().__init__(ast)

    def mutate_BinOp(self, ast_node):
        ast_node.lhs = self.mutate(ast_node.lhs)
        ast_node.rhs = self.mutate(ast_node.rhs)

        if ast_node.is_vector_property_access():
            layout = ast_node.lhs.layout()

            for i in ast_node.vector_indexes():
                flat_index = None

                if layout == Layout_AoS:
                    flat_index = ast_node.rhs * ast_node.sim.dimensions + i

                elif layout == Layout_SoA:
                    flat_index = i * ast_node.sim.particle_capacity + ast_node.rhs

                else:
                    raise Exception("Invalid property layout!")

                ast_node.map_vector_index(i, flat_index)

        return ast_node


def flatten_property_accesses(ast_node):
    flatten = FlattenPropertyAccesses(ast_node)
    flatten.mutate()
