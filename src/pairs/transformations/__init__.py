from pairs.analysis import Analysis
from pairs.transformations.blocks import MergeAdjacentBlocks
from pairs.transformations.devices import AddDeviceCopies, AddDeviceKernels
from pairs.transformations.expressions import ReplaceSymbols, SimplifyExpressions, PrioritizeScalarOps
from pairs.transformations.loops import LICM
from pairs.transformations.lower import Lower
from pairs.transformations.modules import DereferenceWriteVariables, AddResizeLogic, ReplaceModulesByCalls


class Transformations:
    def __init__(self, ast, target):
        self._ast = ast
        self._target = target
        self._analysis = Analysis(ast)
        self._merge_adjacent_blocks = MergeAdjacentBlocks(ast)
        self._replace_symbols = ReplaceSymbols(ast)
        self._simplify_expressions = SimplifyExpressions(ast)
        self._prioritize_scalar_ops = PrioritizeScalarOps(ast)
        self._licm = LICM(ast)
        self._dereference_write_variables = DereferenceWriteVariables(ast)
        self._add_resize_logic = AddResizeLogic(ast)
        self._replace_modules_by_calls = ReplaceModulesByCalls(ast)

        if target.is_gpu():
            self._add_device_copies = AddDeviceCopies(ast)
            self._add_device_kernels = AddDeviceKernels(ast)

    def lower_everything(self):
        nlowered = 1
        while nlowered > 0:
            lower = Lower(self._ast)
            lower.mutate()
            nlowered = lower.lowered_nodes

        self._merge_adjacent_blocks.mutate()

    def optimize_expressions(self):
        self._replace_symbols.mutate()
        self._simplify_expressions.mutate()
        self._prioritize_scalar_ops.mutate()
        self._simplify_expressions.mutate()
        self._analysis.set_used_bin_ops()

    def licm(self):
        self._analysis.set_parent_block()
        self._analysis.set_block_variants()
        self._analysis.set_bin_op_terminals()
        self._licm.mutate()

    def modularize(self):
        self._add_resize_logic.mutate()
        self._analysis.fetch_modules_references()
        self._dereference_write_variables.mutate()
        self._replace_modules_by_calls.set_module_resizes(self._add_resize_logic.module_resizes)
        self._replace_modules_by_calls.mutate()
        self._merge_adjacent_blocks.mutate()

    def add_device_copies(self):
        if self._target.is_gpu():
            self._add_device_copies.mutate()

    def add_device_kernels(self):
        if self._target.is_gpu():
            self._add_device_kernels.mutate()
            self._analysis.fetch_kernel_references()

    def apply_all(self):
        self.lower_everything()
        self.optimize_expressions()
        self.licm()
        self.modularize()
        self.add_device_copies()
        self.add_device_kernels()
