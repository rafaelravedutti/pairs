class Visitor:
    def __init__(self, ast, enter_fn=None, leave_fn=None, max_depth=0):
        self.ast = ast
        self.enter_fn = enter_fn
        self.leave_fn = leave_fn
        self.max_depth = max_depth

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

    def yield_elements(ast, depth, max_depth):
        yield ast
        if depth < max_depth or max_depth == 0:
            for child in ast.children():
                for child_node in Visitor.yield_elements(child, depth + 1, max_depth):
                    yield child_node

    def __iter__(self):
        yield from Visitor.yield_elements(self.ast, 0, self.max_depth)
