from ast.bin_op import ASTTerm
from ast.data_types import Type_Int


class Sizeof(ASTTerm):
    def __init__(self, sim, data_type):
        super().__init__(sim)
        self.data_type = data_type

    def type(self):
        return Type_Int
