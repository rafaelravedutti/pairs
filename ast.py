from part_prot import produced_stmts

expression_id = 0

class BlockAST:
    def __init__(self, stmts, block_type):
        self.stmts = stmts
        self.block_type = block_type

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
        return ExprAST(self, other, '+')

    def __sub__(self, other):
        return ExprAST(self, other, '-')

    def __mul__(self, other):
        return ExprAST(self, other, '*')

    def __rmul__(self, other):
        return ExprAST(other, self, '*')

    def __truediv__(self, other):
        return ExprAST(self, other, '/')

    def __rtruediv__(self, other):
        return ExprAST(other, self, '/')

    def __lt__(self, other):
        return ExprAST(self, other, '<')

    def set(self, other):
        global produced_stmts
        if self.mem is False:
            assert("Invalid assignment: lvalue expected!")
        else:
            produced_stmts.append(AssignAST(self, other))

    def add(self, other):
        global produced_stmts
        if self.mem is False:
            assert("Invalid assignment: lvalue expected!")
        else:
            produced_stmts.append(AssignAST(self, self + other))

class AssignAST:
    def __init__(self, dest, src):
        self.dest = dest
        self.src = src

class IfAST:
    def __init__(self, cond, block_if, block_else):
        self.cond = cond
        self.block_if = block_if
        self.block_else = block_else

class IterAST:
    pass

class NbIterAST:
    pass
