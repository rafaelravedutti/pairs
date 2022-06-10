from pairs.ir.bin_op import ASTTerm
from pairs.ir.lit import Lit
from pairs.ir.types import Types


class Cast(ASTTerm):
    def __init__(self, sim, expr, cast_type):
        super().__init__(sim)
        self.expr = Lit.cvt(sim, expr)
        self.cast_type = cast_type

    def __str__(self):
        return f"Cast<{self.expr}, {self.cast_type}>"

    def int(sim, expr):
        return Cast(sim, expr, Types.Int32)

    def uint64(sim, expr):
        return Cast(sim, expr, Types.UInt64)

    def double(sim, expr):
        return Cast(sim, expr, Types.Double)

    def type(self):
        return self.cast_type

    def children(self):
        return [self.expr]
