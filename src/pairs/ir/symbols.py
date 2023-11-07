from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.lit import Lit
from pairs.ir.operator_class import OperatorClass
from pairs.ir.types import Types


class Symbol(ASTTerm):
    def __init__(self, sim, sym_type):
        super().__init__(sim, OperatorClass.from_type(sym_type))
        self.sym_type = sym_type
        self.assign_to = None

    def __str__(self):
        return f"Symbol<{Types.c_keyword(self.sim, self.sym_type)}>"

    def assign(self, node):
        self.assign_to = node

    def type(self):
        return self.sym_type

    def __getitem__(self, index):
        return SymbolAccess(self.sim, self, Lit.cvt(self.sim, index))


class SymbolAccess(ASTTerm):
    def __init__(self, sim, symbol, index):
        assert not Types.is_scalar(symbol.type()), "Scalar symbols cannot be indexed."
        super().__init__(sim, OperatorClass.from_type(symbol.type()))
        self._symbol = symbol
        self._index = index
        symbol.add_index_to_generate(index)

    def __str__(self):
        return f"SymbolAccess<{self._symbol}, {self._index}>"

    def symbol(self):
        return self._symbol

    def index(self):
        return self._index

    def type(self):
        if not Types.is_scalar(self._symbol.type()):
            return Types.Real

        return self._symbol.type()
