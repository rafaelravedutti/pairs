class Visitor:
    def __init__(self, ast, enter_fn, leave_fn):
        self.ast = ast
        self.enter_fn = enter_fn
        self.leave_fn = leave_fn

    def visit(self):
        self.visit_rec(self.ast)

    def visit_rec(self, ast):
        if self.enter_fn is not None:
            self.enter_fn(ast)

        for c in ast.children():
            self.visit_rec(c)

        if self.leave_fn is not None:
            self.leave_fn(ast)

    def list_ast(self):
        self.list_elements(self.ast)

    def list_elements(self, ast):
        ast_list = [ast]

        for c in self.ast.children():
            ast_list += self.list_elements(c)

        return ast_list

    def yield_elements(self, ast):
        yield ast

        for c in ast.children():
            self.yield_elements(c)

    def __iter__(self):
        self.yield_elements(self.ast)
