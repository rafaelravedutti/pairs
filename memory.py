from printer import printer

class ReallocAST:
    def __init__(self, array, size):
        self.array = array
        self.size = size

    def generate(self, mem=False):
        return printer.print(f"{self.array.name()} = realloc({self.size.generate()})")

    def transform(self, fn):
        self.array = self.array.transform(fn)
        self.size = self.size.transform(fn)
        return fn(self)
