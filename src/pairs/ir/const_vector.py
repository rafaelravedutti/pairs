from pairs.ir.ast_node import ASTNode
from pairs.ir.lit import Lit
from pairs.ir.types import Types


class ConstVector(ASTNode):
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

    def __getitem__(self, expr_ast):
        if isinstance(expr_ast, Lit) or isinstance(expr_ast, int):
            return self.values[expr_ast]

        if self.array is None:
            self.array = self.sim.add_static_array(f"cv{self.const_vector_id}", self.sim.ndims(), Types.Double)

        return self_array[expr_ast]


class ZeroVector(ASTNode):
    def __init__(self, sim):
        super().__init__(sim)

    def __str__(self):
        return f"ZeroVector<>"

    def type(self):
        return Types.Vector

    def __getitem__(self, expr_ast):
        # This allows out-of-bound access to the vector, which may not be good.
        # It is however difficult to evaluate this possibilty without performance
        # loss when the expression value is only known at runtime
        return Lit(self.sim, 0.0)
