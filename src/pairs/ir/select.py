from pairs.ir.accessor_class import AccessorClass
from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.lit import Lit
from pairs.ir.operator_class import OperatorClass
from pairs.ir.types import Types


class Select(ASTTerm):
    last_select = 0

    def new_id():
        Select.last_select += 1
        return Select.last_select - 1

    def __init__(self, sim, cond, expr_if, expr_else):
        super().__init__(sim, OperatorClass.from_type(Lit.cvt(sim, expr_if).type()))
        self.cond = Lit.cvt(sim, cond)
        self.expr_if = Lit.cvt(sim, expr_if)
        self.expr_else = Lit.cvt(sim, expr_else)
        self.select_id = Select.new_id()
        #self.expr_if = ScalarOp.inline(Lit.cvt(sim, expr_if))
        #self.expr_else = ScalarOp.inline(Lit.cvt(sim, expr_else))
        self.terminals = set()
        self.inlined = False
        assert self.expr_if.type() == self.expr_else.type(), "Select: expressions must be of same type!"

    def __str__(self):
        return f"Select<{self.cond}, {self.expr_if}, {self.expr_else}>"

    def __getitem__(self, index):
        _acc_class = AccessorClass.from_type(self.expr_if.type())
        return _acc_class(self.sim, self, Lit.cvt(self.sim, index))

    def id(self):
        return self.select_id

    def name(self):
        return f"sel{self.id()}" + self.label_suffix()

    def type(self):
        return self.expr_if.type()

    def inline_recursively(self):
        method_name = "inline_recursively"
        self.inlined = True

        if hasattr(self.cond, method_name) and callable(getattr(self.cond, method_name)):
            self.cond.inline_recursively()

        if hasattr(self.expr_if, method_name) and callable(getattr(self.expr_if, method_name)):
            self.expr_if.inline_recursively()

        if hasattr(self.expr_else, method_name) and callable(getattr(self.expr_else, method_name)):
            self.expr_else.inline_recursively()

        return self

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.cond, self.expr_if, self.expr_else]
