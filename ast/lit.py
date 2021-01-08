from ast.ast_node import ASTNode
from ast.data_types import Type_Invalid, Type_Int, Type_Float, Type_Bool, Type_Vector


def is_literal(a):
    return isinstance(a, (int, float, bool, list))


def as_lit_ast(sim, a):
    return Lit(sim, a) if is_literal(a) else a


class Lit(ASTNode):
    def __init__(self, sim, value):
        super().__init__(sim)
        self.value = value
        self.lit_type = Type_Invalid

        if isinstance(value, int):
            self.lit_type = Type_Int

        if isinstance(value, float):
            self.lit_type = Type_Float

        if isinstance(value, bool):
            self.lit_type = Type_Bool

        if isinstance(value, list):
            self.lit_type = Type_Vector

        assert self.lit_type != Type_Invalid, "Invalid literal type!"

    def __str__(self):
        return f"Lit<{self.value}>"

    def __eq__(self, other):
        if isinstance(other, Lit):
            return self.value == other.value

        return self.value == other

    def __req__(self, other):
        return self.__cmp__(other)

    def type(self):
        return self.lit_type
