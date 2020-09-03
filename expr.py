from assign import AssignAST
from data_types import Type_Int, Type_Float, Type_Vector
from lit import is_literal, LitAST
from loops import IterAST
from printer import printer
from properties import Property

def is_expr(e):
    return isinstance(e, ExprAST) or isinstance(e, IterAST) or isinstance(e, LitAST)

class ExprAST:
    def __init__(self, sim, lhs, rhs, op, mem=False):
        self.sim = sim
        self.expr_id = sim.new_expr()
        self.lhs = lhs if not is_literal(lhs) else LitAST(lhs)
        self.rhs = rhs if not is_literal(rhs) else LitAST(rhs)
        self.op = op
        self.mem = mem
        self.expr_type = ExprAST.infer_type(self.lhs, self.rhs, self.op)
        self.generated = False

    def __str__(self):
        return "Expr<a: {}, b: {}, op: {}>".format(self.lhs, self.rhs, self.op)

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

    def inv(self):
        return ExprAST(self.sim, 1.0, self, '/')

    def __getitem__(self, index):
        assert self.lhs.type() == Type_Vector, "Cannot use operator [] on specified type!"
        return ExprAST(self.sim, self, index, '[]', self.mem)

    def set(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        self.sim.produced_stmts.append(AssignAST(self.sim, self, other))

    def add(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        self.sim.produced_stmts.append(AssignAST(self.sim, self, self + other))

    def infer_type(lhs, rhs, op):
        lhs_type = lhs.type()
        rhs_type = rhs.type()

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

    def indexed(self, index):
        vname = self.generate()
        if self.expr_type == Type_Vector:
            return f"{vname}[{index}]" if self.mem else f"{vname}_{index}"

        return vname

    def generate(self, mem=False):
        lexpr = self.lhs.generate(mem)
        rexpr = self.rhs.generate()
        if self.op == '[]':
            return f"{lexpr}[{rexpr}]" if self.mem else f"{lexpr}_{rexpr}"

        vname = f"v{self.expr_id}"
        if self.generated is False:
            if self.expr_type == Type_Vector:
                for i in range(0, self.sim.dimensions):
                    li = lexpr if not isinstance(self.lhs, ExprAST) else self.lhs.indexed(i)
                    ri = rexpr if not isinstance(self.rhs, ExprAST) else self.rhs.indexed(i)
                    printer.print(f"double {vname}_{i} = {li} {self.op} {ri};")
            else:
                t = 'double' if self.expr_type == Type_Float else 'int'
                printer.print(f"{t} {vname} = {lexpr} {self.op} {rexpr};")

            self.generated = True

        return vname
