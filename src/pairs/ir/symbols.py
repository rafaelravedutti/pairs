from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.assign import Assign
from pairs.ir.scalars import ScalarOp
from pairs.ir.lit import Lit
from pairs.ir.types import Types
from pairs.ir.vectors import VectorAccess, VectorOp


class Symbol(ASTTerm):
    def __init__(self, sim, sym_type):
        super().__init__(sim, ScalarOp if sym_type != Types.Vector else VectorOp)
        self.sym_type = sym_type
        self.assign_to = None

    def __str__(self):
        return f"Symbol<{Types.c_keyword(self.sym_type)}>"

    def assign(self, node):
        self.assign_to = node

    def type(self):
        return self.sym_type

    def __getitem__(self, index):
        return VectorAccess(self.sim, self, Lit.cvt(self.sim, index))
