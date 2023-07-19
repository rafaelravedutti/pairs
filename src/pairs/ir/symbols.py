from pairs.ir.ast_node import ASTNode
from pairs.ir.assign import Assign
from pairs.ir.bin_op import ASTTerm 
from pairs.ir.types import Types


class Symbol(ASTTerm):
    def __init__(self, sim, sym_type):
        super().__init__(sim)
        self.sym_type = sym_type
        self.assign_to = None

    def __str__(self):
        return f"Symbol<{self.sym_type}>"

    def assign(self, node):
        self.assign_to = node

    def type(self):
        return self.sym_type

    def __getitem__(self, index):
        return SymbolAccess(self.sim, self, index)


class SymbolAccess(ASTTerm):
    def __init__(self, sim, symbol, indexes):
        super().__init__(sim)
        self._symbol = symbol
        self._indexes = indexes if isinstance(indexes, list) else [indexes]
        assert symbol.type() == Types.Vector, "Only vector symbols can be indexed!"

    def __str__(self):
        return f"SymbolAccess<{self._symbol, self._indexes}>"

    def symbol(self):
        return self._symbol

    def indexes(self):
        return self._indexes

    def type(self):
        return self._symbol.type()
