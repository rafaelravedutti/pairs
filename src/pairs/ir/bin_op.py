from pairs.ir.ast_node import ASTNode
from pairs.ir.assign import Assign
from pairs.ir.lit import Lit
from pairs.ir.types import Types
from pairs.ir.vector_expr import VectorExpression


class Decl(ASTNode):
    def __init__(self, sim, elem):
        super().__init__(sim)
        self.elem = elem
        self.used = not sim.check_decl_usage
        sim.add_statement(self)

    def __str__(self):
        return f"Decl<self.elem>"

    def children(self):
        return [self.elem]


class BinOp(VectorExpression):
    last_bin_op = 0

    def new_id():
        BinOp.last_bin_op += 1
        return BinOp.last_bin_op - 1

    def inline(op):
        if hasattr(op, "inline_rec") and callable(getattr(op, "inline_rec")):
            return op.inline_rec()

        return op

    def __init__(self, sim, lhs, rhs, op, mem=False):
        super().__init__(sim)
        self.bin_op_id = BinOp.new_id()
        self.lhs = Lit.cvt(sim, lhs)
        self.rhs = Lit.cvt(sim, rhs)
        self.op = op
        self.mem = mem
        self.inlined = False
        self.generated = False
        self.bin_op_type = BinOp.infer_type(self.lhs, self.rhs, self.op)
        self.terminals = set()
        self.decl = Decl(sim, self)

    def reassign(self, lhs, rhs, op):
        assert self.generated is False, "Error on reassign: BinOp {} already generated!".format(self.bin_op_id)
        self.lhs = Lit.cvt(self.sim, lhs)
        self.rhs = Lit.cvt(self.sim, rhs)
        self.op = op
        self.bin_op_type = BinOp.infer_type(self.lhs, self.rhs, self.op)

    def __str__(self):
        a = self.lhs.id() if isinstance(self.lhs, BinOp) else self.lhs
        b = self.rhs.id() if isinstance(self.rhs, BinOp) else self.rhs
        return f"BinOp<{a} {self.op} {b}>"

    def match(self, bin_op):
        return self.lhs == bin_op.lhs and self.rhs == bin_op.rhs and self.op == bin_op.operator()

    def x(self):
        return self.__getitem__(0)

    def y(self):
        return self.__getitem__(1)

    def z(self):
        return self.__getitem__(2)

    def set(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        return self.sim.add_statement(Assign(self.sim, self, other))

    def add(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        return self.sim.add_statement(Assign(self.sim, self, self + other))

    def sub(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        return self.sim.add_statement(Assign(self.sim, self, self - other))

    def infer_type(lhs, rhs, op):
        lhs_type = lhs.type()
        rhs_type = rhs.type()

        if op in ['>', '<', '>=', '<=', '==', '!=']:
            return Types.Boolean

        if op == '[]':
            if lhs_type == Types.Vector:
                return Types.Double

            return lhs_type

        if lhs_type == rhs_type:
            return lhs_type

        if Types.is_integer(lhs_type) or Types.is_integer(rhs_type):
            if isinstance(lhs, Lit) or Lit.is_literal(lhs):
                return rhs_type

            if isinstance(rhs, Lit) or Lit.is_literal(rhs):
                return lhs_type

            # TODO: Are more checkings required here to generate proper integer data type?
            return lhs_type

        if lhs_type == Types.Vector or rhs_type == Types.Vector:
            return Types.Vector

        if Types.is_real(lhs_type) or Types.is_real(rhs_type):
            return Types.Double

        return None

    def inline_rec(self):
        self.inlined = True

        if hasattr(self.lhs, "inline_rec") and callable(getattr(self.lhs, "inline_rec")):
            self.lhs.inline_rec()

        if hasattr(self.rhs, "inline_rec") and callable(getattr(self.rhs, "inline_rec")):
            self.rhs.inline_rec()

        return self

    def id(self):
        return self.bin_op_id

    def type(self):
        return self.bin_op_type

    def declaration(self):
        return self.decl

    def operator(self):
        return self.op

    def add_terminal(self, terminal):
        self.terminals.add(terminal)

    def children(self):
        return [self.lhs, self.rhs] + list(super().children())

    def __getitem__(self, index):
        super().__getitem__(index)
        return VectorAccess(self.sim, self, Lit.cvt(self.sim, index))

    def __add__(self, other):
        return BinOp(self.sim, self, other, '+')

    def __radd__(self, other):
        return BinOp(self.sim, other, self, '+')

    def __sub__(self, other):
        return BinOp(self.sim, self, other, '-')

    def __mul__(self, other):
        return BinOp(self.sim, self, other, '*')

    def __rmul__(self, other):
        return BinOp(self.sim, other, self, '*')

    def __truediv__(self, other):
        return BinOp(self.sim, self, other, '/')

    def __rtruediv__(self, other):
        return BinOp(self.sim, other, self, '/')

    def __lt__(self, other):
        return BinOp(self.sim, self, other, '<')

    def __le__(self, other):
        return BinOp(self.sim, self, other, '<=')

    def __gt__(self, other):
        return BinOp(self.sim, self, other, '>')

    def __ge__(self, other):
        return BinOp(self.sim, self, other, '>=')

    def and_op(self, other):
        return BinOp(self.sim, self, other, '&&')

    def cmp(lhs, rhs):
        return BinOp(lhs.sim, lhs, rhs, '==')

    def neq(lhs, rhs):
        return BinOp(lhs.sim, lhs, rhs, '!=')

    def inv(self):
        return BinOp(self.sim, 1.0, self, '/')

    def __mod__(self, other):
        return BinOp(self.sim, self, other, '%')


class ASTTerm(ASTNode):
    def __init__(self, sim):
        super().__init__(sim)

    def __add__(self, other):
        return BinOp(self.sim, self, other, '+')

    def __radd__(self, other):
        return BinOp(self.sim, other, self, '+')

    def __sub__(self, other):
        return BinOp(self.sim, self, other, '-')

    def __mul__(self, other):
        return BinOp(self.sim, self, other, '*')

    def __rmul__(self, other):
        return BinOp(self.sim, other, self, '*')

    def __truediv__(self, other):
        return BinOp(self.sim, self, other, '/')

    def __rtruediv__(self, other):
        return BinOp(self.sim, other, self, '/')

    def __lt__(self, other):
        return BinOp(self.sim, self, other, '<')

    def __le__(self, other):
        return BinOp(self.sim, self, other, '<=')

    def __gt__(self, other):
        return BinOp(self.sim, self, other, '>')

    def __ge__(self, other):
        return BinOp(self.sim, self, other, '>=')

    def __and__(self, other):
        return BinOp(self.sim, self, other, '&')

    def __or__(self, other):
        return BinOp(self.sim, self, other, '|')

    def __xor__(self, other):
        return BinOp(self.sim, self, other, '^')

    def __invert__(self):
        return BinOp(self.sim, self, None, '~')

    def and_op(self, other):
        return BinOp(self.sim, self, other, '&&')

    def or_op(self, other):
        return BinOp(self.sim, self, other, '||')

    def cmp(lhs, rhs):
        return BinOp(lhs.sim, lhs, rhs, '==')

    def neq(lhs, rhs):
        return BinOp(lhs.sim, lhs, rhs, '!=')

    def inv(self):
        return BinOp(self.sim, 1.0, self, '/')

    def __mod__(self, other):
        return BinOp(self.sim, self, other, '%')


class VectorAccess(ASTTerm):
    def __init__(self, sim, expr, index):
        super().__init__(sim)
        self.expr = expr
        self.index = index

    def type(self):
        return Types.Double

    def set(self, other):
        return self.sim.add_statement(Assign(self.sim, self, other))

    def add(self, other):
        return self.sim.add_statement(Assign(self.sim, self, self + other))

    def sub(self, other):
        return self.sim.add_statement(Assign(self.sim, self, self - other))

    def children(self):
        return [self.expr]
