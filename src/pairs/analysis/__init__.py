import time
from pairs.analysis.bin_ops import ResetInPlaceBinOps, SetBinOpTerminals, SetInPlaceBinOps, SetDeclaredExprs
from pairs.analysis.blocks import SetBlockVariants, SetExprOwnerBlock, SetParentBlock
from pairs.analysis.devices import FetchKernelReferences
from pairs.analysis.modules import FetchModulesReferences


class Analysis:
    def __init__(self, ast):
        self._ast = ast

    def apply(self, analysis):
        print(f"Performing analysis: {type(analysis).__name__}... ", end="")
        start = time.time()
        analysis.set_ast(self._ast)
        analysis.visit()
        elapsed = time.time() - start
        print(f"{elapsed:.2f}s elapsed.")

    def set_bin_op_terminals(self):
        self.apply(SetBinOpTerminals())

    def set_block_variants(self):
        SetBlockVariants(self._ast).mutate()

    def set_parent_block(self):
        self.apply(SetParentBlock())

    def set_expressions_owner_block(self):
        set_expr_owner_block = SetExprOwnerBlock()
        self.apply(set_expr_owner_block)
        return (set_expr_owner_block.ownership, set_expr_owner_block.expressions_to_lift)

    def fetch_kernel_references(self):
        self.apply(ResetInPlaceBinOps())
        self.apply(SetInPlaceBinOps())
        self.apply(FetchKernelReferences())

    def fetch_modules_references(self):
        self.apply(FetchModulesReferences())

    def set_declared_expressions(self):
        set_decl_exprs = SetDeclaredExprs()
        self.apply(set_decl_exprs)
        return set_decl_exprs.declared_exprs
