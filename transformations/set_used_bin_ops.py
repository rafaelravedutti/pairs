from ir.bin_op import BinOp
from ir.visitor import Visitor


class SetUsedBinOps(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.bin_ops = []

    def visit_BinOpDef(self, ast_node):
        pass

    def visit_BinOp(self, ast_node):
        ast_node.bin_op_def.used = True
        self.visit_children(ast_node)
        # TODO: These expressions could be automatically included in visitor traversal
        for vidxs in ast_node.mapped_expressions():
            self.visit(vidxs)


def set_used_bin_ops(ast):
    set_used_binops = SetUsedBinOps(ast)
    set_used_binops.visit()
