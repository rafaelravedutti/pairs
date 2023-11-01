from pairs.ir.ast_term import ASTTerm
from pairs.ir.types import Types


class Lit(ASTTerm):
    def is_literal(a):
        return isinstance(a, (int, float, bool, str, list))

    def cvt(sim, a):
        return Lit(sim, a) if Lit.is_literal(a) else a

    def __init__(self, sim, value):
        if isinstance(value, list):
            non_scalar_mapping = {
                sim.ndims(): Types.Vector,
                sim.ndims() * sim.ndims(): Types.Matrix,
                sim.ndims() + 1: Types.Quaternion
            }

            self.lit_type = non_scalar_mapping.get(len(value), Types.Invalid)

        else:
            scalar_mapping = {
                int: Types.Int32,
                float: Types.Real,
                bool: Types.Boolean,
                str: Types.String,
            }

            self.lit_type = scalar_mapping.get(type(value), Types.Invalid)

        assert self.lit_type != Types.Invalid, "Invalid literal type."
        from pairs.ir.operator_class import OperatorClass
        super().__init__(sim, OperatorClass.from_type(self.lit_type))
        self.value = value

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
