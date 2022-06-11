from pairs.analysis import Analysis
from pairs.transformations.blocks import LiftExprOwnerBlocks, MergeAdjacentBlocks
from pairs.transformations.devices import AddDeviceCopies, AddDeviceKernels
from pairs.transformations.expressions import ReplaceSymbols, SimplifyExpressions, PrioritizeScalarOps
from pairs.transformations.loops import LICM
from pairs.transformations.lower import Lower
from pairs.transformations.modules import DereferenceWriteVariables, AddResizeLogic, ReplaceModulesByCalls


class Transformations:
    def __init__(self, ast, target):
        self._ast = ast
        self._target = target

    def apply(self, transformation, data=None):
        transformation.set_ast(self._ast)
        if data is not None:
            transformation.set_data(data)

        self._ast = transformation.mutate()

    def analysis(self):
        return Analysis(self._ast)

    def lower(self, lower_finals=False):
        nlowered = 1
        while nlowered > 0:
            lower = Lower()
            self.apply(lower, [lower_finals])
            nlowered = lower.lowered_nodes

        self.apply(MergeAdjacentBlocks())

    def optimize_expressions(self):
        self.apply(ReplaceSymbols())
        self.apply(SimplifyExpressions())
        self.apply(PrioritizeScalarOps())
        self.apply(SimplifyExpressions())
        self.analysis().set_used_bin_ops()

    def lift_expressions_to_owner_blocks(self):
        ownership, expressions_to_lift = self.analysis().set_expressions_owner_block()
        self.apply(LiftExprOwnerBlocks(), [ownership, expressions_to_lift])

    def licm(self):
        self.analysis().set_parent_block()
        self.analysis().set_block_variants()
        self.analysis().set_bin_op_terminals()
        self.apply(LICM())

    def modularize(self):
        add_resize_logic = AddResizeLogic()
        self.apply(add_resize_logic)
        self.analysis().fetch_modules_references()
        self.apply(DereferenceWriteVariables())
        self.apply(ReplaceModulesByCalls(), [add_resize_logic.module_resizes])
        self.apply(MergeAdjacentBlocks())

    def add_device_copies(self):
        if self._target.is_gpu():
            self.apply(AddDeviceCopies())

    def add_device_kernels(self):
        if self._target.is_gpu():
            self.apply(AddDeviceKernels())
            self.analysis().fetch_kernel_references()

    def apply_all(self):
        self.lower()
        self.optimize_expressions()
        self.lift_expressions_to_owner_blocks()
        self.licm()
        self.modularize()
        self.add_device_copies()
        self.add_device_kernels()
        self.lower(True)
