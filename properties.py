expression_id = 0

class Property:
    def __init__(self, prop_name, default_value):
        self.prop_name = prop_name
        self.default_value = default_value

    def __getitem__(self, expr_ast):
        ExprAST(self.prop_name, expr_ast, '[]', true)

class ExprAST:
    def __init__(self, lhs, rhs, op, mem=False):
        global expression_id
        self.expr_id = expression_id
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.mem = mem
        expression_id += 1

    def __add__(self, other):
        ExprAST(self, other, '+')

    def __sub__(self, other):
        ExprAST(self, other, '-')

    def __mul__(self, other):
        ExprAST(self, other, '*')

    def __div__(self, other):
        ExprAST(self, other, '/')

    def set(self, other):
        if self.mem is False:
            assert("Invalid assignment: lvalue expected!")
        else:
            block.append(AssignAST(self, other))

    def add(self, other):
        if self.mem is False:
            assert("Invalid assignment: lvalue expected!")
        else:
            block.append(AssignAST(self, self + other))

class IterAST:
    pass

class NbIterAST
    pass
