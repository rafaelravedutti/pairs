class Array:
    def __init__(self, sim, arr_name, arr_size, arr_type):
        self.sim = sim
        self.arr_name = arr_name
        self.arr_size = arr_size
        self.arr_type = arr_type

    def __str__(self):
        return f"Array<name: {self.arr_name}, size: {self.arr_size}, type: {self.arr_type}>"

    def name(self):
        return self.arr_name

    def size(self):
        return self.arr_size

    def type(self):
        return self.arr_type

    def __getitem__(self, expr_ast):
        from expr import ExprAST
        return ExprAST(self.sim, self, expr_ast, '[]', True)

    def generate(self, mem=False):
        return self.arr_name

    def transform(self, fn):
        return fn(self)
