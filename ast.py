from part_prot import produced_stmts
from block_types import ParticlePairsBlock, ParticlesBlock

expression_id = 0

class BlockAST:
    def __init__(self, stmts, block_type):
        self.stmts = stmts
        self.block_type = block_type

    def generate(self):
        if self.block_type == ParticlePairsBlock:
            print("for particle_pairs:");
        elif self.block_type == ParticlesBlock:
            print("for particles:");
        else:
            raise Exception("Invalid block type!")

        for stmt in self.stmts:
            stmt.generate()

class ExprAST:
    def __init__(self, lhs, rhs, op, mem=False):
        global expression_id
        self.expr_id = expression_id
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.mem = mem
        self.generated = False
        expression_id += 1

    def __str__(self):
        return "Expr <a: {}, b: {}, op: {}>".format(self.lhs, self.rhs, self.op)

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

    def inv(self):
        return ExprAST(1.0, self, '/')

    def set(self, other):
        global produced_stmts
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        produced_stmts.append(AssignAST(self, other))

    def add(self, other):
        global produced_stmts
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        produced_stmts.append(AssignAST(self, self + other))

    def generate(self, mem=False):
        if isinstance(self.lhs, ExprAST):
            lvname = self.lhs.generate(mem)
        else:
            lvname = self.lhs

        if isinstance(self.rhs, ExprAST):
            rvname = self.rhs.generate()
        else:
            rvname = self.rhs

        if self.op == '[]':
            output = "{}[{}]".format(lvname, rvname)
        elif self.op == 'vector_len_sq':
            output = "vector_len_sq({})".format(lvname)
        else:
            output = "{} {} {}".format(lvname, self.op, rvname)

        if mem:
            return output

        vname = "v{}".format(self.expr_id)

        if self.generated is False:
            print("{} = {}".format(vname, output))
            self.generated = True

        return vname

class AssignAST:
    def __init__(self, dest, src):
        self.dest = dest
        self.src = src
        self.generated = False

    def __str__(self):
        return "Assign<a: {}, b: {}>".format(dest, src)

    def generate(self):
        if self.generated is False:
            print("{} = {}".format(self.dest.generate(True), self.src.generate()))
            self.generated = True

class IfAST:
    def __init__(self, cond, block_if, block_else):
        self.cond = cond
        self.block_if = block_if
        self.block_else = block_else

    def generate(self):
        cvname = self.cond.generate()
        print("if({}) {{".format(cvname))

        for stmt in self.block_if:
            stmt.generate()

        if self.block_else is not None:
            print("} else {")
            for stmt in self.block_else:
                stmt.generate()

        print("}")
        
class IterAST:
    def __str__(self):
        return "IterAST <>"

class NbIterAST:
    def __str__(self):
        return "NbIterAST <>"
