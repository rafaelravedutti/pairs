from data_types import Type_Vector
from lit import is_literal, LitAST
from printer import printer

class AssignAST:
    def __init__(self, dest, src):
        self.dest = dest
        self.src = src if not is_literal(src) else LitAST(src)
        self.generated = False

    def __str__(self):
        return "Assign<a: {}, b: {}>".format(dest, src)

    def generate(self):
        if self.generated is False:
            if self.dest.expr_type == Type_Vector:
                d = self.dest.generate(True)
                s = self.src.generate()
                printer.print("{}[0] = {}_0;".format(d, s))
                printer.print("{}[1] = {}_1;".format(d, s))
                printer.print("{}[2] = {}_2;".format(d, s))

            else:
                printer.print("{} = {};".format(self.dest.generate(True), self.src.generate()))

            self.generated = True
