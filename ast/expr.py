from ast.assign import AssignAST
from ast.data_types import Type_Int, Type_Float, Type_Bool, Type_Vector
from ast.lit import is_literal, LitAST
from ast.loops import IterAST
from ast.properties import Property
from code_gen.printer import printer

class ExprAST:
    def __init__(self, sim, lhs, rhs, op, mem=False):
        self.sim = sim
        self.expr_id = sim.new_expr()
        self.lhs = lhs if not is_literal(lhs) else LitAST(lhs)
        self.rhs = rhs if not is_literal(rhs) else LitAST(rhs)
        self.op = op
        self.mem = mem
        self.expr_type = ExprAST.infer_type(self.lhs, self.rhs, self.op)
        self.vec_generated = []
        self.generated = False

    def __str__(self):
        return f"Expr<a: {self.lhs}, b: {self.rhs}, op: {self.op}>"

    def __add__(self, other):
        return ExprAST(self.sim, self, other, '+')

    def __radd__(self, other):
        return ExprAST(self.sim, other, self, '+')

    def __sub__(self, other):
        return ExprAST(self.sim, self, other, '-')

    def __mul__(self, other):
        return ExprAST(self.sim, self, other, '*')

    def __rmul__(self, other):
        return ExprAST(self.sim, other, self, '*')

    def __truediv__(self, other):
        return ExprAST(self.sim, self, other, '/')

    def __rtruediv__(self, other):
        return ExprAST(self.sim, other, self, '/')

    def __lt__(self, other):
        return ExprAST(self.sim, self, other, '<')

    def __le__(self, other):
        return ExprAST(self.sim, self, other, '<=')

    def __gt__(self, other):
        return ExprAST(self.sim, self, other, '>')

    def __ge__(self, other):
        return ExprAST(self.sim, self, other, '>=')

    def and_op(self, other):
        return ExprAST(self.sim, self, other, '&&')

    def cmp(lhs, rhs):
        return ExprAST(lhs.sim, lhs, rhs, '==')

    def inv(self):
        return ExprAST(self.sim, 1.0, self, '/')

    def __getitem__(self, index):
        assert self.lhs.type() == Type_Vector, "Cannot use operator [] on specified type!"
        index_ast = index if not is_literal(index) else LitAST(index)
        return ExprVecAST(self.sim, self, index_ast)

    def generated_vector_index(self, index):
        return not [i for i in self.vec_generated if i == index]

    def set(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        return self.sim.capture_statement(AssignAST(self.sim, self, other))

    def add(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        return self.sim.capture_statement(AssignAST(self.sim, self, self + other))

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

    def generate(self, mem=False):
        lexpr = self.lhs.generate(mem)
        rexpr = self.rhs.generate()
        if self.op == '[]':
            return f"{lexpr}[{rexpr}]" if self.mem else f"{lexpr}_{rexpr}"

        ename = f"e{self.expr_id}"
        if self.generated is False:
            assert self.expr_type != Type_Vector, "Vector code must be generated through ExprVecAST class!"
            t = 'double' if self.expr_type == Type_Float else 'int' if self.expr_type == Type_Int else 'bool'
            printer.print(f"const {t} {ename} = {lexpr} {self.op} {rexpr};")
            self.generated = True

        return ename

    def generate_inline(self, mem=False):
        lexpr = self.lhs.generate_inline(mem) if isinstance(self.lhs, ExprAST) else self.lhs.generate(mem)
        rexpr = self.rhs.generate_inline() if isinstance(self.rhs, ExprAST) else self.rhs.generate()

        if self.op == '[]':
            return f"{lexpr}[{rexpr}]" if self.mem else f"{lexpr}_{rexpr}"

        assert self.expr_type != Type_Vector, "Vector code must be generated through ExprVecAST class!"
        return f"{lexpr} {self.op} {rexpr}"

    def transform(self, fn):
        self.lhs = self.lhs.transform(fn)
        self.rhs = self.rhs.transform(fn)
        return fn(self)

class ExprVecAST():
    def __init__(self, sim, expr, index):
        self.sim = sim
        self.expr = expr
        self.index = index
        self.lhs = expr.lhs if not isinstance(expr.lhs, ExprAST) else ExprVecAST(sim, expr.lhs, index)
        self.rhs = expr.rhs if not isinstance(expr.rhs, ExprAST) else ExprVecAST(sim, expr.rhs, index)

    def __str__(self):
        return f"ExprVecAST<a: {self.lhs}, b: {self.rhs}, op: {self.expr.op}, i: {self.index}>"

    def __sub__(self, other):
        return ExprAST(self.sim, self, other, '-')

    def __mul__(self, other):
        return ExprAST(self.sim, self, other, '*')

    def idx(self):
        return self.index

    def type(self):
        return Type_Float

    def generate(self, mem=False):
        if self.expr.type() != Type_Vector:
            return self.expr.generate()

        iexpr = self.index.generate()
        if self.expr.op == '[]':
            expr = self.expr.generate()
            return f"{expr}[{iexpr}]"

        ename = f"e{self.expr.expr_id}[{iexpr}]" if self.expr.mem else f"e{self.expr.expr_id}_{iexpr}"
        if self.expr.generated_vector_index(iexpr):
            lexpr = self.lhs.generate(mem)
            rexpr = self.rhs.generate()
            printer.print(f"const double {ename} = {lexpr} {self.expr.op} {rexpr};")
            self.expr.vec_generated.append(iexpr)

        return ename

    def transform(self, fn):
        self.lhs = self.lhs.transform(fn)
        self.rhs = self.rhs.transform(fn)
        self.index = self.index.transform(fn)
        return fn(self)