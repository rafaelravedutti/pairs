from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.lit import Lit
from pairs.ir.types import Types


class VectorOp(ASTTerm):
    last_vector_op = 0

    def new_id():
        VectorOp.last_vector_op += 1
        return VectorOp.last_vector_op - 1

    def __init__(self, sim, lhs, rhs, op, mem=False):
        assert lhs.type() == Types.Vector or rhs.type() == Types.Vector, \
            "VectorOp(): At least one vector operand is required."
        super().__init__(sim, VectorOp)
        self._id = VectorOp.new_id()
        self.lhs = Lit.cvt(sim, lhs)
        self.rhs = Lit.cvt(sim, rhs)
        self.op = op
        self.mem = mem
        self.in_place = False
        self.terminals = set()

    def __str__(self):
        a = f"VectorOp<{self.lhs.id()}>" if isinstance(self.lhs, VectorOp) else self.lhs
        b = f"VectorOp<{self.rhs.id()}>" if isinstance(self.rhs, VectorOp) else self.rhs
        return f"VectorOp<id={self.id()}, {a} {self.op.symbol()} {b}>"

    def __getitem__(self, index):
        return VectorAccess(self.sim, self, Lit.cvt(self.sim, index))

    def id(self):
        return self._id

    def type(self):
        return Types.Vector

    def operator(self):
        return self.op

    def reassign(self, lhs, rhs, op):
        self.lhs = Lit.cvt(self.sim, lhs)
        self.rhs = Lit.cvt(self.sim, rhs)
        self.op = op

    def copy(self):
        return VectorOp(self.sim, self.lhs.copy(), self.rhs.copy(), self.op, self.mem)

    def match(self, vector_op):
        return self.lhs == vector_op.lhs and self.rhs == vector_op.rhs and self.op == vector_op.operator()

    def x(self):
        return self.__getitem__(0)

    def y(self):
        return self.__getitem__(1)

    def z(self):
        return self.__getitem__(2)

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.lhs, self.rhs] if not self.op.is_unary() else [self.lhs]


class VectorAccess(ASTTerm):
    def __init__(self, sim, expr, index):
        super().__init__(sim, ScalarOp)
        self.expr = expr
        self.index = index
        expr.add_index_to_generate(index)

    def __str__(self):
        return f"VectorAccess<{self.expr}, {self.index}>"

    def type(self):
        return Types.Double

    def children(self):
        return [self.expr]


class Vector(ASTTerm):
    last_vector = 0

    def new_id():
        Vector.last_vector += 1
        return Vector.last_vector - 1

    def __init__(self, sim, values):
        assert isinstance(values, list) and len(values) == sim.ndims(), "Vector(): Given list is invalid!"
        super().__init__(sim, VectorOp)
        self._id = Vector.new_id()
        self._values = [Lit.cvt(sim, v) for v in values]
        self.terminals = set()

    def __str__(self):
        return f"Vector<{self._values}>"

    def __getitem__(self, index):
        return VectorAccess(self.sim, self, Lit.cvt(self.sim, index))

    def id(self):
        return self._id

    def type(self):
        return Types.Vector

    def get_value(self, dimension):
        return self._values[dimension]

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return self._values


class ZeroVector(ASTTerm):
    def __init__(self, sim):
        super().__init__(sim, VectorOp)

    def __str__(self):
        return f"ZeroVector<>"

    def __getitem__(self, expr_ast):
        # This allows out-of-bound access to the vector, which may not be good.
        # It is however difficult to evaluate this possibilty without performance
        # loss when the expression value is only known at runtime
        return Lit(self.sim, 0.0)

    def type(self):
        return Types.Vector

    def children(self):
        return []
