from pairs.ir.ast_term import ASTTerm
from pairs.ir.scalars import ScalarOp
from pairs.ir.types import Types


class MathFunction(ASTTerm):
    last_math_func = 0

    def new_id():
        MathFunction.last_math_func += 1
        return MathFunction.last_math_func

    def __init__(self, sim):
        super().__init__(sim, ScalarOp)
        self._id = MathFunction.new_id()
        self._params = []
        self.terminals = set()
        self.inlined = False

    def __str__(self):
        return f"MathFunction<{self.function_name(), self.parameters()}>"

    def id(self):
        return self._id

    def name(self):
        return f"mf{self.id()}" + self.label_suffix()

    def function_name(self):
        return "undefined"

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

    def parameters(self):
        return self._params

    def children(self):
        return self._params

class Sqrt(MathFunction):
    def __init__(self, sim, expr):
        super().__init__(sim)
        self._params = [expr]

    def __str__(self):
        return f"Sqrt<{self._params}>"

    def function_name(self):
        return "sqrt" if self.sim.use_double_precision() else "sqrtf"

    def type(self):
        return self._params[0].type()


class Abs(MathFunction):
    def __init__(self, sim, expr):
        super().__init__(sim)
        self._params = [expr]

    def __str__(self):
        return f"Abs<{self._params}>"

    def function_name(self):
        return "fabs"

    def type(self):
        return self._params[0].type()


class Sin(MathFunction):
    def __init__(self, sim, expr):
        super().__init__(sim)
        self._params = [expr]

    def __str__(self):
        return f"Sin<{self._params}>"

    def function_name(self):
        return "sin" if self.sim.use_double_precision() else "sinf"

    def type(self):
        return self._params[0].type()


class Cos(MathFunction):
    def __init__(self, sim, expr):
        super().__init__(sim)
        self._params = [expr]

    def __str__(self):
        return f"Cos<{self._params}>"

    def function_name(self):
        return "cos" if self.sim.use_double_precision() else "cosf"

    def type(self):
        return self._params[0].type()


class Ceil(MathFunction):
    def __init__(self, sim, expr):
        assert Types.is_real(expr.type()), "Expression must be of real type!"
        super().__init__(sim)
        self._params = [expr]

    def __str__(self):
        return f"Ceil<{self._params}>"

    def function_name(self):
        return "ceil" if self.sim.use_double_precision() else "ceilf"

    def type(self):
        return Types.Int32

