from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.lit import Lit
from pairs.ir.types import Types


class QuaternionOp(ASTTerm):
    last_quaternion_op = 0

    def new_id():
        QuaternionOp.last_quaternion_op += 1
        return QuaternionOp.last_quaternion_op - 1

    def __init__(self, sim, lhs, rhs, op, mem=False):
        assert lhs.type() == Types.Quaternion or rhs.type() == Types.Quaternion, \
            "QuaternionOp(): At least one quaternion operand is required."
        super().__init__(sim, QuaternionOp)
        self._id = QuaternionOp.new_id()
        self.lhs = Lit.cvt(sim, lhs)
        self.rhs = Lit.cvt(sim, rhs)
        self.op = op
        self.mem = mem
        self.in_place = False
        self.terminals = set()

    def __str__(self):
        a = f"QuaternionOp<{self.lhs.id()}>" if isinstance(self.lhs, QuaternionOp) else self.lhs
        b = f"QuaternionOp<{self.rhs.id()}>" if isinstance(self.rhs, QuaternionOp) else self.rhs
        return f"QuaternionOp<id={self.id()}, {a} {self.op.symbol()} {b}>"

    def __getitem__(self, index):
        return QuaternionAccess(self.sim, self, Lit.cvt(self.sim, index))

    def id(self):
        return self._id

    def name(self):
        return f"quat_op{self.id()}" + self.label_suffix()

    def type(self):
        return Types.Quaternion

    def operator(self):
        return self.op

    def reassign(self, lhs, rhs, op):
        self.lhs = Lit.cvt(self.sim, lhs)
        self.rhs = Lit.cvt(self.sim, rhs)
        self.op = op

    def copy(self):
        return QuaternionOp(self.sim, self.lhs.copy(), self.rhs.copy(), self.op, self.mem)

    def match(self, quat_op):
        return self.lhs == quat_op.lhs and self.rhs == quat_op.rhs and self.op == quat_op.operator()

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.lhs, self.rhs] if not self.op.is_unary() else [self.lhs]


class QuaternionAccess(ASTTerm):
    def __init__(self, sim, expr, index):
        super().__init__(sim, ScalarOp)
        self.expr = expr
        self.index = index
        expr.add_index_to_generate(index)

    def __str__(self):
        return f"QuaternionAccess<{self.expr}, {self.index}>"

    def type(self):
        return Types.Double

    def children(self):
        return [self.expr]


class Quaternion(ASTTerm):
    last_quat = 0

    def new_id():
        Quaternion.last_quat += 1
        return Quaternion.last_quat - 1

    def __init__(self, sim, values):
        assert isinstance(values, list) and len(values) == sim.ndims() + 1, \
            "Quaternion(): Given list is invalid!"
        super().__init__(sim, QuaternionOp)
        self._id = Quaternion.new_id()
        self._values = [Lit.cvt(sim, v) for v in values]
        self.terminals = set()

    def __str__(self):
        return f"Quaternion<{self._values}>"

    def __getitem__(self, index):
        return QuaternionAccess(self.sim, self, Lit.cvt(self.sim, index))

    def id(self):
        return self._id

    def name(self):
        return f"quat{self.id()}" + self.label_suffix()

    def type(self):
        return Types.Quaternion

    def get_value(self, dimension):
        return self._values[dimension]

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return self._values
