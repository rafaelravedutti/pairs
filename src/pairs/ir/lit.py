from pairs.ir.ast_node import ASTNode
from pairs.ir.types import Types


class Lit(ASTNode):
    def is_literal(a):
        return isinstance(a, (int, float, bool, str, list))

    def cvt(sim, a):
        return Lit(sim, a) if Lit.is_literal(a) else a

    def __init__(self, sim, value):
        super().__init__(sim)
        self.value = value

        type_mapping = {
            int: Types.Int32,
            float: Types.Double,
            bool: Types.Boolean,
            str: Types.String,
            list: Types.Vector
        }

        self.lit_type = type_mapping.get(type(value), Types.Invalid)
        assert self.lit_type != Types.Invalid, "Invalid literal type!"

    def __str__(self):
        return f"Lit<{self.value}>"

    def __eq__(self, other):
        if isinstance(other, Lit):
            return self.value == other.value

        return self.value == other

    def __req__(self, other):
        return self.__cmp__(other)

    def copy(self):
        return Lit(self.sim, self.value)

    def type(self):
        return self.lit_type
