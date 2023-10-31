from pairs.ir.ast_node import ASTNode
from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.lit import Lit
from pairs.ir.types import Types


class MatrixOp(ASTTerm):
    last_matrix_op = 0

    def new_id():
        MatrixOp.last_matrix_op += 1
        return MatrixOp.last_matrix_op - 1

    def __init__(self, sim, lhs, rhs, op, mem=False):
        assert lhs.type() == Types.Matrix or rhs.type() == Types.Matrix, \
            "MatrixOp(): At least one matrix operand is required."
        super().__init__(sim, MatrixOp)
        self._id = MatrixOp.new_id()
        self.lhs = Lit.cvt(sim, lhs)
        self.rhs = Lit.cvt(sim, rhs)
        self.op = op
        self.mem = mem
        self.in_place = False
        self.terminals = set()

    def __str__(self):
        a = f"MatrixOp<{self.lhs.id()}>" if isinstance(self.lhs, MatrixOp) else self.lhs
        b = f"MatrixOp<{self.rhs.id()}>" if isinstance(self.rhs, MatrixOp) else self.rhs
        return f"MatrixOp<id={self.id()}, {a} {self.op.symbol()} {b}>"

    def __getitem__(self, index):
        return MatrixAccess(self.sim, self, Lit.cvt(self.sim, index))

    def id(self):
        return self._id

    def name(self):
        return f"mat_op{self.id()}" + self.label_suffix()

    def type(self):
        return Types.Matrix

    def operator(self):
        return self.op

    def reassign(self, lhs, rhs, op):
        self.lhs = Lit.cvt(self.sim, lhs)
        self.rhs = Lit.cvt(self.sim, rhs)
        self.op = op

    def copy(self):
        return MatrixOp(self.sim, self.lhs.copy(), self.rhs.copy(), self.op, self.mem)

    def match(self, matrix_op):
        return self.lhs == matrix_op.lhs and self.rhs == matrix_op.rhs and self.op == matrix_op.operator()

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.lhs, self.rhs] if not self.op.is_unary() else [self.lhs]


class MatrixAccess(ASTTerm):
    def __init__(self, sim, expr, index):
        super().__init__(sim, ScalarOp)
        self.expr = expr
        self.index = index
        expr.add_index_to_generate(index)

    def __str__(self):
        return f"MatrixAccess<{self.expr}, {self.index}>"

    def type(self):
        return Types.Double

    def children(self):
        return [self.expr]


class Matrix(ASTTerm):
    last_matrix = 0

    def new_id():
        Matrix.last_matrix += 1
        return Matrix.last_matrix - 1

    def __init__(self, sim, values):
        assert isinstance(values, list) and len(values) == sim.ndims() * sim.ndims(), \
            "Matrix(): Given list is invalid!"
        super().__init__(sim, MatrixOp)
        self._id = Matrix.new_id()
        self._values = [Lit.cvt(sim, v) for v in values]
        self.terminals = set()

    def __str__(self):
        return f"Matrix<{self._values}>"

    def __getitem__(self, index):
        return MatrixAccess(self.sim, self, Lit.cvt(self.sim, index))

    def id(self):
        return self._id

    def name(self):
        return f"mat{self.id()}" + self.label_suffix()

    def type(self):
        return Types.Matrix

    def get_value(self, dimension):
        return self._values[dimension]

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return self._values
