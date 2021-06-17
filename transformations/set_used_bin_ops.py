from ir.bin_op import BinOp
from ir.visitor import Visitor


class SetUsedBinOps(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.bin_ops = []

    def visit_BinOp(self, ast_node):
        ast_node.bin_op_def.used = True


def set_used_bin_ops(ast):
    set_used_binops = SetUsedBinOps(ast)
    set_used_binops.visit()
