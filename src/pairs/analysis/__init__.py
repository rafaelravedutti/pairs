from pairs.analysis.bin_ops import ResetInPlaceBinOps, SetBinOpTerminals, SetInPlaceBinOps, SetUsedBinOps
from pairs.analysis.blocks import SetBlockVariants, SetExprOwnerBlock, SetParentBlock
from pairs.analysis.devices import FetchKernelReferences
from pairs.analysis.modules import FetchModulesReferences


class Analysis:
    def __init__(self, ast):
        self._ast = ast
        self._set_used_bin_ops = SetUsedBinOps(ast)
        self._set_bin_op_terminals = SetBinOpTerminals(ast)
        self._set_block_variants = SetBlockVariants(ast)
        self._set_parent_block = SetParentBlock(ast)
        self._set_expressions_owner_block = SetExprOwnerBlock(ast)
        self._reset_in_place_bin_ops = ResetInPlaceBinOps(ast)
        self._set_in_place_bin_ops = SetInPlaceBinOps(ast)
        self._fetch_kernel_references = FetchKernelReferences(ast)
        self._fetch_modules_references = FetchModulesReferences(ast)

    def set_used_bin_ops(self):
        self._set_used_bin_ops.visit()

    def set_bin_op_terminals(self):
        self._set_bin_op_terminals.visit()

    def set_block_variants(self):
        self._set_block_variants.mutate()

    def set_parent_block(self):
        self._set_parent_block.visit()

    def set_expressions_owner_block(self):
        self._set_expressions_owner_block.visit()
        return (self._set_expressions_owner_block.ownership, self._set_expressions_owner_block.expressions_to_lift)

    def fetch_kernel_references(self):
        self._reset_in_place_bin_ops.visit()
        self._set_in_place_bin_ops.visit()
        self._fetch_kernel_references.visit()

    def fetch_modules_references(self):
        self._fetch_modules_references.visit()
