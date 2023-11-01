from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.lit import Lit
from pairs.ir.operators import Operators
from pairs.ir.types import Types


class ScalarOp(ASTTerm):
    last_scalar_op = 0

    def new_id():
        ScalarOp.last_scalar_op += 1
        return ScalarOp.last_scalar_op - 1

    def inline(op):
        method_name = "inline_recursively"
        if hasattr(op, method_name) and callable(getattr(op, method_name)):
            return op.inline_recursively()

        return op

    def cmp(lhs, rhs):
        return ScalarOp(lhs.sim, lhs, rhs, Operators.Eq)

    def neq(lhs, rhs):
        return ScalarOp(lhs.sim, lhs, rhs, Operators.Neq)

    def __init__(self, sim, lhs, rhs, op, mem=False):
        super().__init__(sim, ScalarOp)
        self.scalar_op_id = ScalarOp.new_id()
        self.lhs = Lit.cvt(sim, lhs)
        self.rhs = Lit.cvt(sim, rhs)
        self.op = op
        self.mem = mem
        self.inlined = False
        self.in_place = False
        self.scalar_op_type = ScalarOp.infer_type(self.lhs, self.rhs, self.op)
        self.terminals = set()

    def reassign(self, lhs, rhs, op):
        self.lhs = Lit.cvt(self.sim, lhs)
        self.rhs = Lit.cvt(self.sim, rhs)
        self.op = op
        self.scalar_op_type = ScalarOp.infer_type(self.lhs, self.rhs, self.op)

    def __str__(self):
        a = f"ScalarOp<{self.lhs.id()}>" if isinstance(self.lhs, ScalarOp) else self.lhs
        b = f"ScalarOp<{self.rhs.id()}>" if isinstance(self.rhs, ScalarOp) else self.rhs
        return f"ScalarOp<id={self.id()}, {a} {self.op.symbol()} {b}>"

    def copy(self):
        return ScalarOp(self.sim, self.lhs.copy(), self.rhs.copy(), self.op, self.mem)

    def match(self, scalar_op):
        return self.lhs == scalar_op.lhs and self.rhs == scalar_op.rhs and self.op == scalar_op.operator()

    def x(self):
        return self.__getitem__(0)

    def y(self):
        return self.__getitem__(1)

    def z(self):
        return self.__getitem__(2)

    def infer_type(lhs, rhs, op):
        lhs_type = lhs.type()

        if op.is_unary():
            return lhs_type

        rhs_type = rhs.type()

        if op.is_conditional():
            return Types.Boolean

        if lhs_type == rhs_type:
            return lhs_type

        if lhs_type == Types.Vector or rhs_type == Types.Vector:
            return Types.Vector

        if lhs_type == Types.Matrix or rhs_type == Types.Matrix:
            return Types.Matrix

        if lhs_type == Types.Quaternion or rhs_type == Types.Quaternion:
            return Types.Quaternion

        if Types.is_real(lhs_type) or Types.is_real(rhs_type):
            return Types.Real

        if Types.is_integer(lhs_type) or Types.is_integer(rhs_type):
            if isinstance(lhs, Lit) or Lit.is_literal(lhs):
                return rhs_type

            if isinstance(rhs, Lit) or Lit.is_literal(rhs):
                return lhs_type

            # TODO: Are more checkings required here to generate proper integer data type?
            return lhs_type

        return None

    def inline_recursively(self):
        method_name = "inline_recursively"
        self.inlined = True

        if hasattr(self.lhs, method_name) and callable(getattr(self.lhs, method_name)):
            self.lhs.inline_recursively()

        if hasattr(self.rhs, method_name) and callable(getattr(self.rhs, method_name)):
            self.rhs.inline_recursively()

        return self

    def id(self):
        return self.scalar_op_id

    def name(self):
        return f"sca_op{self.id()}" + self.label_suffix()

    def type(self):
        return self.scalar_op_type

    def operator(self):
        return self.op

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.lhs, self.rhs] if not self.op.is_unary() else [self.lhs]
