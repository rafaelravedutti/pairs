from printer import printer
from properties import Property
from block_types import ParticlePairsBlock, ParticlesBlock

def is_expr(e):
    return isinstance(e, ExprAST) or isinstance(e, IterAST) or isinstance(e, NbIterAST)

def get_expr_type(expr):
    if expr is None:
        return None

    if isinstance(expr, ExprAST):
        return expr.expr_type

    if isinstance(expr, int) or isinstance(expr, IterAST) or isinstance(expr, NbIterAST):
        return 'integer'

    if isinstance(expr, float):
        return 'real'

    if isinstance(expr, Property):
        return expr.prop_type

    return None

def infer_expr_type(lhs_type, rhs_type, op):
    if op == 'vector_len_sq':
        return 'real'

    if op == '[]':
        return lhs_type

    if lhs_type == rhs_type:
        return lhs_type

    if lhs_type == 'vector' or rhs_type == 'vector':
        return 'vector'

    if lhs_type == 'real' or rhs_type == 'real':
        return 'real'

    return None

def suffixed(var_name, index, var_type):
    if var_type != 'vector' or isinstance(var_name, str) is False:
        return var_name

    if var_name[-1] == ']':
        return var_name + '[{}]'.format(index)

    return var_name + '_{}'.format(index)

class BlockAST:
    def __init__(self, stmts, block_type):
        self.stmts = stmts
        self.block_type = block_type

    def generate(self):
        if self.block_type == ParticlePairsBlock:
            printer.print("for(int i = 0; i < nparticles; ++i) {");
            printer.add_ind(4)
            printer.print("for(int j = 0; j < nparticles; ++j) {");
            printer.add_ind(4)
            printer.print("if(i != j) {");
            printer.add_ind(4)
            nclose = 3
        elif self.block_type == ParticlesBlock:
            printer.print("for(int i = 0; i < nparticles; ++i) {");
            printer.add_ind(4)
            nclose = 1
        else:
            raise Exception("Invalid block type!")

        for stmt in self.stmts:
            stmt.generate()

        for _ in range(0, nclose):
            printer.add_ind(-4)
            printer.print("}")

class ExprAST:
    def __init__(self, sim, lhs, rhs, op, mem=False):
        self.sim = sim
        self.expr_id = sim.new_expr()
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.mem = mem
        self.lhs_type = get_expr_type(lhs)
        self.rhs_type = get_expr_type(rhs)
        self.expr_type = infer_expr_type(self.lhs_type, self.rhs_type, self.op)
        self.generated = False

    def __str__(self):
        return "Expr <a: {}, b: {}, op: {}>".format(self.lhs, self.rhs, self.op)

    def __add__(self, other):
        return ExprAST(self.sim, self, other, '+')

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

    def set(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        self.sim.produced_stmts.append(AssignAST(self, other))

    def add(self, other):
        assert self.mem is True, "Invalid assignment: lvalue expected!"
        self.sim.produced_stmts.append(AssignAST(self, self + other))

    def generate(self, mem=False):
        if is_expr(self.lhs):
            lvname = self.lhs.generate(mem)
        elif isinstance(self.lhs, Property):
            lvname = self.lhs.prop_name
        else:
            lvname = self.lhs

        if is_expr(self.rhs):
            rvname = self.rhs.generate()
        elif isinstance(self.rhs, Property):
            rvname = self.rhs.prop_name
        else:
            rvname = self.rhs

        if self.op == '[]':
            output = "{}[{}]".format(lvname, rvname)
        elif self.op == 'vector_len_sq':
            terms = []
            for i in range(0, 3):
                t = suffixed(lvname, i, self.lhs_type)
                terms.append("{} * {}".format(t, t))
            output = "{} + {} + {}".format(terms[0], terms[1], terms[2])
        else:
            output = "{} {} {}".format(lvname, self.op, rvname)

        if mem:
            return output

        vname = "v{}".format(self.expr_id)

        if self.generated is False:
            if self.expr_type == 'vector':
                for i in range(0, 3):
                    if self.op == '[]':
                        output = suffixed("{}[{}]".format(lvname, rvname), i, self.lhs_type)
                    else:
                        output = "{} {} {}".format(suffixed(lvname, i, self.lhs_type), self.op, suffixed(rvname, i, self.rhs_type))

                    printer.print("double {} = {};".format(suffixed(vname, i, self.expr_type), output))
            else:
                t = 'double' if self.expr_type == 'real' else 'int'
                printer.print("{} {} = {};".format(t, vname, output))

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
            if self.src.expr_type == 'vector':
                d = self.dest.generate(True)
                s = self.src.generate()
                printer.print("{}[0] = {}_0;".format(d, s))
                printer.print("{}[1] = {}_1;".format(d, s))
                printer.print("{}[2] = {}_2;".format(d, s))

            else:
                printer.print("{} = {};".format(self.dest.generate(True), self.src.generate()))

            self.generated = True

class IfAST:
    def __init__(self, cond, block_if, block_else):
        self.cond = cond
        self.block_if = block_if
        self.block_else = block_else

    def generate(self):
        cvname = self.cond.generate()
        printer.print("if({}) {{".format(cvname))

        printer.add_ind(4)
        for stmt in self.block_if:
            stmt.generate()
        printer.add_ind(-4)

        if self.block_else is not None:
            printer.print("} else {")
            printer.add_ind(4)
            for stmt in self.block_else:
                stmt.generate()
            printer.add_ind(-4)

        printer.print("}")
        
class IterAST:
    def __str__(self):
        return "IterAST <>"

    def generate(self):
        return 'i'

class NbIterAST:
    def __str__(self):
        return "NbIterAST <>"

    def generate(self):
        return 'j'
