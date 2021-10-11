from pairs.ir.visitor import Visitor


class SetUsedBinOps(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.bin_ops = []

    def visit_BinOp(self, ast_node):
        ast_node.decl.used = True
        self.visit_children(ast_node)

    def visit_Decl(self, ast_node):
        pass

    def visit_PropertyAccess(self, ast_node):
        ast_node.decl.used = True
        self.visit_children(ast_node)

def set_used_bin_ops(ast):
    set_used_binops = SetUsedBinOps(ast)
    set_used_binops.visit()
