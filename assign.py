from data_types import Type_Vector
from lit import is_literal, LitAST
from printer import printer

class AssignAST:
    def __init__(self, sim, dest, src):
        self.sim = sim
        self.dest = dest
        self.src = src if not is_literal(src) else LitAST(src)
        self.generated = False

    def __str__(self):
        return f"Assign<a: {dest}, b: {src}>"

    def generate(self):
        if self.generated is False:
            d = self.dest.generate(True)
            if self.dest.type() == Type_Vector:
                for i in range(0, self.sim.dimensions):
                    from expr import ExprAST
                    si = self.src.generate() if not isinstance(self.src, ExprAST) or self.src.type() != Type_Vector else self.src[i].generate()
                    printer.print(f"{d}[{i}] = {si};")

            else:
                s = self.src.generate()
                printer.print(f"{d} = {s};")

            self.generated = True
