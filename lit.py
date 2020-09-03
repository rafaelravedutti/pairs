from data_types import Type_Invalid, Type_Int, Type_Float, Type_Vector

def is_literal(a):
    return isinstance(a, int) or isinstance(a, float) or isinstance(a, list)

class LitAST:
    def __init__(self, value):
        self.value = value
        self.lit_type = Type_Invalid

        if isinstance(value, int):
            self.lit_type = Type_Int

        if isinstance(value, float):
            self.lit_type = Type_Float

        if isinstance(value, list):
            self.lit_type = Type_Vector

        assert self.lit_type != Type_Invalid, "Invalid literal type!"

    def __str__(self):
        return f"Lit<{self.value}>"

    def type(self):
        return self.lit_type

    def generate(self, mem=False):
        assert mem is False, "Literal is not lvalue!"
        return self.value
