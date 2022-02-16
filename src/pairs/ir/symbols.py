from pairs.ir.ast_node import ASTNode
from pairs.ir.assign import Assign
from pairs.ir.bin_op import ASTTerm 


class Symbol(ASTTerm):
    def __init__(self, sim, sym_type):
        super().__init__(sim)
        self.sym_type = sym_type
        self.assign_to = None

    def __str__(self):
        return f"Symbol<{self.var_type}>"

    def assign(self, node):
        self.assign_to = node

    def type(self):
        return self.sym_type
