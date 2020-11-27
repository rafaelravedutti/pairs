from ast.assign import Assign
from ast.data_types import Type_Float, Type_Bool, Type_Vector
from ast.lit import as_lit_ast
from ast.properties import Property


class Expr:
    last_expr = 0

    def new_id():
        Expr.last_expr += 1
        return Expr.last_expr - 1

    def __init__(self, sim, lhs, rhs, op, mem=False):
        self.sim = sim
        self.expr_id = Expr.new_id()
        self.lhs = as_lit_ast(sim, lhs)
        self.rhs = as_lit_ast(sim, rhs)
        self.op = op
        self.mem = mem
        self.expr_type = Expr.infer_type(self.lhs, self.rhs, self.op)
        self.expr_scope = None
        self.vec_generated = []
        self.generated = False

    def __str__(self):
        return f"Expr<a: {self.lhs}, b: {self.rhs}, op: {self.op}>"

    def __add__(self, other):
        return Expr(self.sim, self, other, '+')

    def __radd__(self, other):
        return Expr(self.sim, other, self, '+')

    def __sub__(self, other):
        return Expr(self.sim, self, other, '-')

    def __mul__(self, other):
        return Expr(self.sim, self, other, '*')

    def __rmul__(self, other):
        return Expr(self.sim, other, self, '*')

    def __truediv__(self, other):
        return Expr(self.sim, self, other, '/')

    def __rtruediv__(self, other):
        return Expr(self.sim, other, self, '/')

    def __lt__(self, other):
        return Expr(self.sim, self, other, '<')

    def __le__(self, other):
        return Expr(self.sim, self, other, '<=')

    def __gt__(self, other):
        return Expr(self.sim, self, other, '>')

    def __ge__(self, other):
        return Expr(self.sim, self, other, '>=')

    def and_op(self, other):
        return Expr(self.sim, self, other, '&&')

    def cmp(lhs, rhs):
        return Expr(lhs.sim, lhs, rhs, '==')

    def neq(lhs, rhs):
        return Expr(lhs.sim, lhs, rhs, '!=')

    def inv(self):
        return Expr(self.sim, 1.0, self, '/')

    def match(self, expr):
        return self.lhs == expr.lhs and \
               self.rhs == expr.rhs and \
               self.op == expr.op

    def x(self):
        return self.__getitem__(0)

    def y(self):
        return self.__getitem__(1)

    def z(self):
        return self.__getitem__(2)

    def __getitem__(self, index):
        assert self.lhs.type() == Type_Vector, \
            "Cannot use operator [] on specified type!"
        return ExprVec(self.sim, self, as_lit_ast(self.sim, index))

    def generated_vector_index(self, index):
        return not [i for i in self.vec_generated if i == index]

    def set(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        return self.sim.add_statement(Assign(self.sim, self, other))

    def add(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        return self.sim.add_statement(Assign(self.sim, self, self + other))

    def infer_type(lhs, rhs, op):
        lhs_type = lhs.type()
        rhs_type = rhs.type()

        if op in ['>', '<', '>=', '<=', '==', '!=']:
            return Type_Bool

        if op == '[]':
            if isinstance(lhs, Property):
                return lhs_type

            if lhs_type == Type_Vector:
                return Type_Float

            return lhs_type

        if lhs_type == rhs_type:
            return lhs_type

        if lhs_type == Type_Vector or rhs_type == Type_Vector:
            return Type_Vector

        if lhs_type == Type_Float or rhs_type == Type_Float:
            return Type_Float

        return None

    def type(self):
        return self.expr_type

    def scope(self):
        if self.expr_scope is None:
            lhs_scp = self.lhs.scope()
            rhs_scp = self.rhs.scope()
            self.expr_scope = lhs_scp if lhs_scp > rhs_scp else rhs_scp

        return self.expr_scope

    def children(self):
        return [self.lhs, self.rhs]

    def generate(self, mem=False):
        lhs_expr = self.lhs.generate(mem)
        rhs_expr = self.rhs.generate()
        if self.op == '[]':
            return self.sim.code_gen.generate_expr_access(
                lhs_expr, rhs_expr, self.mem)

        if self.generated is False:
            assert self.expr_type != Type_Vector, \
                "Vector code must be generated through ExprVec class!"

            self.sim.code_gen.generate_expr(
                self.expr_id, self.expr_type, lhs_expr, rhs_expr, self.op)
            self.generated = True

        return self.sim.code_gen.generate_expr_ref(self.expr_id)

    def generate_inline(self, mem=False, recursive=False):
        inline_lhs_expr = recursive and isinstance(self.lhs, Expr)
        inline_rhs_expr = recursive and isinstance(self.rhs, Expr)
        lhs_expr = (self.lhs.generate_inline(recursive, mem) if inline_lhs_expr
                    else self.lhs.generate(mem))
        rhs_expr = (self.rhs.generate_inline(recursive) if inline_rhs_expr
                    else self.rhs.generate())

        if self.op == '[]':
            return self.sim.code_gen.generate_expr_access(
                lhs_expr, rhs_expr, self.mem)

        assert self.expr_type != Type_Vector, \
            "Vector code must be generated through ExprVec class!"
        return self.sim.code_gen.generate_inline_expr(
                lhs_expr, rhs_expr, self.op)

    def transform(self, fn):
        self.lhs = self.lhs.transform(fn)
        self.rhs = self.rhs.transform(fn)
        return fn(self)


class ExprVec():
    def __init__(self, sim, expr, index):
        self.sim = sim
        self.expr = expr
        self.index = index
        self.expr_scope = None
        self.lhs = (expr.lhs if not isinstance(expr.lhs, Expr)
                    else ExprVec(sim, expr.lhs, index))
        self.rhs = (expr.rhs if not isinstance(expr.rhs, Expr)
                    else ExprVec(sim, expr.rhs, index))

    def __str__(self):
        return (f"ExprVec<a: {self.lhs}, b: {self.rhs}, " +
                f"op: {self.expr.op} i: {self.index}>")

    def __add__(self, other):
        return Expr(self.sim, self, other, '+')

    def __radd__(self, other):
        return Expr(self.sim, other, self, '+')

    def __sub__(self, other):
        return Expr(self.sim, self, other, '-')

    def __mul__(self, other):
        return Expr(self.sim, self, other, '*')

    def __lt__(self, other):
        return Expr(self.sim, self, other, '<')

    def __le__(self, other):
        return Expr(self.sim, self, other, '<=')

    def __gt__(self, other):
        return Expr(self.sim, self, other, '>')

    def __ge__(self, other):
        return Expr(self.sim, self, other, '>=')

    def set(self, other):
        return self.sim.add_statement(Assign(self.sim, self, other))

    def add(self, other):
        return self.sim.add_statement(Assign(self.sim, self, self + other))

    def sub(self, other):
        return self.sim.add_statement(Assign(self.sim, self, self - other))

    def type(self):
        return Type_Float

    def scope(self):
        if self.expr_scope is None:
            expr_scp = self.expr.scope()
            index_scp = self.index.scope()
            self.expr_scope = expr_scp if expr_scp > index_scp else index_scp

        return self.expr_scope

    def children(self):
        return [self.lhs, self.rhs, self.index]

    def generate(self, mem=False):
        if self.expr.type() != Type_Vector:
            return self.expr.generate()

        index_expr = self.index.generate()
        if self.expr.op == '[]':
            return self.sim.code_gen.generate_expr_access(
                self.expr.generate(), index_expr, True)

        if self.expr.generated_vector_index(index_expr):
            self.sim.code_gen.generate_vec_expr(
                self.expr.expr_id,
                index_expr,
                self.lhs.generate(mem),
                self.rhs.generate(),
                self.expr.op,
                self.expr.mem)

            self.expr.vec_generated.append(index_expr)

        return self.sim.code_gen.generate_vec_expr_ref(
            self.expr.expr_id, index_expr, self.expr.mem)

    def transform(self, fn):
        self.lhs = self.lhs.transform(fn)
        self.rhs = self.rhs.transform(fn)
        self.index = self.index.transform(fn)
        return fn(self)
