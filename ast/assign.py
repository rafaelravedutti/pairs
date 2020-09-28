from ast.data_types import Type_Vector
from ast.lit import is_literal, LitAST
from code_gen.printer import printer

class AssignAST:
    def __init__(self, sim, dest, src):
        self.sim = sim
        self.type = dest.type()
        self.generated = False
        src = src if not is_literal(src) else LitAST(src)

        if dest.type() == Type_Vector:
            self.assignments = []

            for i in range(0, sim.dimensions):
                from ast.expr import ExprAST
                self.assignments.append((dest[i], src if not isinstance(src, ExprAST) or src.type() != Type_Vector else src[i]))
        else:
            self.assignments = [(dest, src)]

    def __str__(self):
        return f"Assign<{self.assignments}>"

    def generate(self):
        if self.generated is False:
            for dest, src in self.assignments:
                d = dest.generate(True)
                s = src.generate()
                printer.print(f"{d} = {s};")

            self.generated = True

    def transform(self, fn):
        self.assignments = [(self.assignments[i][0].transform(fn), self.assignments[i][1].transform(fn)) for i in range(0, len(self.assignments))]
        return fn(self)