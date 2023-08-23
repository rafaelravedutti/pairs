from collections import deque
import pairs.ir.utils as util


class Visitor:
    def __init__(self, ast=None, max_depth=0, breadth_first=False):
        self.ast = ast
        self.max_depth = max_depth
        self.breadth_first = breadth_first
        self.visited_nodes = []

    def __iter__(self):
        if self.breadth_first:
            yield from self.yield_elements_breadth_first(self.ast)
        else:
            yield self.ast
            yield from self.yield_elements(self.ast, 1)

    def set_ast(self, ast):
        self.ast = ast

    def clear_visited_nodes(self):
        self.visited_nodes = []

    def get_method(self, method_name):
        method = getattr(self, method_name, None)
        return method if callable(method) else None

    def visit(self, ast_nodes=None):
        if ast_nodes is None:
            ast_nodes = [self.ast]

        if not isinstance(ast_nodes, list):
            ast_nodes = [ast_nodes]

        for node in ast_nodes:
            terminal_node = util.is_terminal(node)
            if terminal_node or node not in self.visited_nodes:
                if not terminal_node:
                    self.visited_nodes.append(node)

                method = self.get_method(f"visit_{type(node).__name__}")
                if method is not None:
                    method(node)
                else:
                    for b in type(node).__bases__:
                        method = self.get_method(f"visit_{b.__name__}")
                        if method is not None:
                            method(node)
                            break

                    if method is None:
                        self.visit(node.children())

    def visit_children(self, ast_node):
        self.visit(ast_node.children())

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
