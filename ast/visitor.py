from collections import deque


class Visitor:
    def __init__(self, ast, max_depth=0, breadth_first=False):
        self.ast = ast
        self.max_depth = max_depth
        self.breadth_first = breadth_first

    def get_method(self, method_name):
        method = getattr(self, method_name, None)
        return method if callable(method) else None

    def visit(self, ast_node=None):
        if ast_node is None:
            ast_node = self.ast

        method = self.get_method(f"visit_{type(ast_node).__name__}")
        if method is not None:
            method(ast_node)
        else:
            self.keep_visiting(ast_node)

    def keep_visiting(self, ast_node):
        for c in ast_node.children():
            self.visit(c)

    def yield_elements_breadth_first(self, ast_node=None):
        nodes_to_visit = deque()

        if ast_node is None:
            ast_node = self.ast

        nodes_to_visit.append(ast_node)

        while nodes_to_visit:
            next_node = nodes_to_visit.popleft() # nodes_to_visit.pop() for depth-first traversal
            yield next_node
            for c in next_node.children():
                nodes_to_visit.append(c)

    def yield_elements(self, ast, depth):
        if depth < self.max_depth or self.max_depth == 0:
            for child in ast.children():
                yield child
                yield from self.yield_elements(child, depth + 1)

    def __iter__(self):
        if self.breadth_first:
            yield from self.yield_elements_breadth_first(self.ast)
        else:
            yield self.ast
            yield from self.yield_elements(self.ast, 1)
