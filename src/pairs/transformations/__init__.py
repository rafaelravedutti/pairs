from pairs.analysis import Analysis
from pairs.transformations.blocks import LiftExprOwnerBlocks, MergeAdjacentBlocks
from pairs.transformations.devices import AddDeviceCopies, AddDeviceKernels, AddHostReferencesToModules, AddDeviceReferencesToModules
from pairs.transformations.expressions import ReplaceSymbols, LowerNeighborIndexes, SimplifyExpressions, PrioritizeScalarOps, AddExpressionDeclarations
from pairs.transformations.loops import LICM
from pairs.transformations.lower import Lower
from pairs.transformations.modules import DereferenceWriteVariables, AddResizeLogic, ReplaceModulesByCalls


class Transformations:
    def __init__(self, ast, target):
        self._ast = ast
        self._target = target
        self._module_resizes = None

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
        self.apply(LowerNeighborIndexes())
        self.apply(SimplifyExpressions())
        self.apply(PrioritizeScalarOps())
        self.apply(SimplifyExpressions())

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
        self._module_resizes = add_resize_logic.module_resizes
        self.analysis().fetch_modules_references()
        self.apply(DereferenceWriteVariables())
        self.apply(ReplaceModulesByCalls(), [self._module_resizes])
        self.apply(MergeAdjacentBlocks())

    def add_device_copies(self):
        if self._target.is_gpu():
            self.apply(AddDeviceCopies(), [self._module_resizes])

    def add_device_kernels(self):
        if self._target.is_gpu():
            self.apply(AddDeviceKernels())
            self.analysis().fetch_kernel_references()

    def add_expression_declarations(self):
        declared_exprs = self.analysis().set_declared_expressions()
        self.apply(AddExpressionDeclarations(), [declared_exprs])

    def add_host_references_to_modules(self):
        if self._target.is_gpu():
            self.apply(AddHostReferencesToModules())

    def add_device_references_to_modules(self):
        if self._target.is_gpu():
            self.apply(AddDeviceReferencesToModules())

    def apply_all(self):
        self.lower()
        self.optimize_expressions()
        self.add_expression_declarations()
        self.lift_expressions_to_owner_blocks()
        self.licm()
        self.modularize()
        self.add_device_kernels()
        self.add_device_copies()
        self.lower(True)
        self.add_expression_declarations()
        self.add_host_references_to_modules()
        self.add_device_references_to_modules()
