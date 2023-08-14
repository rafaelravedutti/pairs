from pairs.ir.bin_op import ASTTerm
from pairs.ir.lit import Lit
from pairs.ir.types import Types
from pairs.ir.vector_expr import VectorExpression


class ConstVector(ASTTerm, VectorExpression):
    def __init__(self, sim, values):
        assert isinstance(values, list) and len(values) == sim.ndims(), \
            "ConstVector(): Given list is invalid!"
        super().__init__(sim)
        self.values = values
        self.array = None

    def __str__(self):
        return f"ConstVector<{self.values}>"

    def type(self):
        return Types.Vector

    def vector_index(self, index):
        if isinstance(index, Lit) or isinstance(index, int):
            return self.values[index]

        if self.array is None:
            self.array = self.sim.add_static_array(f"cv{self.const_vector_id}", self.sim.ndims(), Types.Double)

        return self_array[index]

    def children(self):
        return []

    def __getitem__(self, expr_ast):
        return self.vector_index(expr_ast)


class ZeroVector(ASTTerm, VectorExpression):
    def __init__(self, sim):
        super().__init__(sim)

    def __str__(self):
        return f"ZeroVector<>"

    def type(self):
        return Types.Vector

    def vector_index(self, index):
        # This allows out-of-bound access to the vector, which may not be good.
        # It is however difficult to evaluate this possibilty without performance
        # loss when the expression value is only known at runtime
        return Lit(self.sim, 0.0)

    def children(self):
        return []

    def __getitem__(self, expr_ast):
        return self.vector_index(expr_ast)
