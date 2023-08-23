import time
from pairs.analysis.expressions import DetermineExpressionsTerminals, ResetInPlaceOperations, DetermineInPlaceOperations, ListDeclaredExpressions
from pairs.analysis.blocks import DiscoverBlockVariants, DetermineExpressionsOwnership, DetermineParentBlocks
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

    def determine_expressions_terminals(self):
        self.apply(DetermineExpressionsTerminals())

    def discover_block_variants(self):
        DiscoverBlockVariants(self._ast).mutate()

    def determine_parent_blocks(self):
        self.apply(DetermineParentBlocks())

    def determine_expressions_ownership(self):
        determine_ownership = DetermineExpressionsOwnership()
        self.apply(determine_ownership)
        return (determine_ownership.ownership, determine_ownership.expressions_to_lift)

    def fetch_kernel_references(self):
        self.apply(ResetInPlaceOperations())
        self.apply(DetermineInPlaceOperations())
        self.apply(FetchKernelReferences())

    def fetch_modules_references(self):
        self.apply(FetchModulesReferences())

    def list_declared_expressions(self):
        list_expressions = ListDeclaredExpressions()
        self.apply(list_expressions)
        return list_expressions.declared_exprs
