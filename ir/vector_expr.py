from ir.ast_node import ASTNode
from ir.data_types import Type_Vector
from ir.lit import Lit


class VectorExpression(ASTNode):
    def __init__(self, sim):
        super().__init__(sim)
        self.vector_indexes = set()
        self.expressions = {}

    def vector_expressions(self):
        return self.expressions.values()

    def indexes(self):
        yield from self.vector_indexes

    def get_index_expression(self, index):
        index_value = index.value if isinstance(index, Lit) else index
        return self.expressions[index_value] if index_value in self.expressions else None

    def vector_index(self, v_index):
        return None

    def propagate_vector_index(self, index):
        self.vector_indexes.add(index)
        index_expr = self.vector_index(index)

        if index_expr is not None:
            self.expressions[index] = index_expr

        for p in self.propagate_through():
            if isinstance(p, VectorExpression) and p.is_vector_kind():
                p.propagate_vector_index(index)

    def is_vector_kind(self):
        return self.type() == Type_Vector

    # Default is to propagate through children, but this can be overridden
    def propagate_through(self):
        return self.children()

    def children(self):
        return self.vector_expressions()

    def __getitem__(self, index):
        assert self.type() == Type_Vector, "Cannot use operator [] on specified type!"
        self.propagate_vector_index(index)
        return self
