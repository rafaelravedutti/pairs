from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.types import Types


class Sizeof(ASTTerm):
    def __init__(self, sim, data_type):
        super().__init__(sim, ScalarOp)
        self.data_type = data_type

    def type(self):
        return Types.Int32
