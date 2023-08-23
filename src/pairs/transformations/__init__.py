import time
from pairs.analysis import Analysis
from pairs.transformations.blocks import LiftDeclarations, MergeAdjacentBlocks
from pairs.transformations.devices import AddDeviceCopies, AddDeviceKernels, AddHostReferencesToModules, AddDeviceReferencesToModules
from pairs.transformations.expressions import ReplaceSymbols, LowerNeighborIndexes, SimplifyExpressions, AddExpressionDeclarations
from pairs.transformations.loops import LICM
from pairs.transformations.lower import Lower
from pairs.transformations.modules import DereferenceWriteVariables, AddResizeLogic, ReplaceModulesByCalls


class Transformations:
    def __init__(self, ast, target):
        self._ast = ast
        self._target = target
        self._module_resizes = None

    def apply(self, transformation, data=None):
        print(f"Applying transformation: {type(transformation).__name__}... ", end="")
        start = time.time()
        transformation.set_ast(self._ast)
        if data is not None:
            transformation.set_data(data)

        self._ast = transformation.mutate()
        elapsed = time.time() - start
        print(f"{elapsed:.2f}s elapsed.")

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
        self.apply(LowerNeighborIndexes())
        self.apply(ReplaceSymbols())
        self.apply(SimplifyExpressions())

    def lift_declarations_to_owner_blocks(self):
        #self.analysis().determine_parent_block()
        ownership, expressions_to_lift = self.analysis().determine_expressions_ownership()
        self.apply(LiftDeclarations(), [ownership, expressions_to_lift])

    def licm(self):
        self.analysis().discover_block_variants()
        self.analysis().determine_expressions_terminals()
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
        declared_exprs = self.analysis().list_declared_expressions()
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
        self.lift_declarations_to_owner_blocks()
        self.licm()
        self.modularize()
        self.add_device_kernels()
        self.add_device_copies()
        self.lower(True)
        self.add_expression_declarations()
        self.add_host_references_to_modules()
        self.add_device_references_to_modules()
