from pairs.ir.ast_node import ASTNode
from pairs.ir.operators import Operators
from pairs.ir.types import Types


class ASTTerm(ASTNode):
    def __init__(self, sim, class_type):
        super().__init__(sim)
        self._class_type = class_type
        self._indexes_to_generate = set()

    def __add__(self, other):
        return self._class_type(self.sim, self, other, Operators.Add)

    def __radd__(self, other):
        return self._class_type(self.sim, other, self, Operators.Add)

    def __sub__(self, other):
        return self._class_type(self.sim, self, other, Operators.Sub)

    def __mul__(self, other):
        return self._class_type(self.sim, self, other, Operators.Mul)

    def __rmul__(self, other):
        return self._class_type(self.sim, other, self, Operators.Mul)

    def __truediv__(self, other):
        return self._class_type(self.sim, self, other, Operators.Div)

    def __rtruediv__(self, other):
        return self._class_type(self.sim, other, self, Operators.Div)

    def __lt__(self, other):
        return self._class_type(self.sim, self, other, Operators.Lt)

    def __le__(self, other):
        return self._class_type(self.sim, self, other, Operators.Leq)

    def __gt__(self, other):
        return self._class_type(self.sim, self, other, Operators.Gt)

    def __ge__(self, other):
        return self._class_type(self.sim, self, other, Operators.Geq)

    def __and__(self, other):
        return self._class_type(self.sim, self, other, Operators.BinAnd)

    def __or__(self, other):
        return self._class_type(self.sim, self, other, Operators.BinOr)

    def __xor__(self, other):
        return self._class_type(self.sim, self, other, Operators.BinXor)

    def __invert__(self):
        return self._class_type(self.sim, self, None, Operators.Invert)

    def __mod__(self, other):
        return self._class_type(self.sim, self, other, Operators.Mod)

    def not_op(self):
        return self._class_type(self.sim, self, None, Operators.Not)

    def and_op(self, other):
        return self._class_type(self.sim, self, other, Operators.And)

    def or_op(self, other):
        return self._class_type(self.sim, self, other, Operators.Or)

    def inv(self):
        return self._class_type(self.sim, 1.0, self, Operators.Div)

    def is_vector(self):
        return self.type() == Types.Vector

    def indexes_to_generate(self):
        return self._indexes_to_generate

    def add_index_to_generate(self, index):
        integer_index = index if isinstance(index, int) else index.value
        assert isinstance(integer_index, int), "add_index_to_generate(): Index must be an integer."
        self._indexes_to_generate.add(integer_index)

        for child in self.children():
            if isinstance(child, ASTTerm) and child.type() == Types.Vector:
                child.add_index_to_generate(integer_index)
