from ast.bin_op import BinOp
from ast.loops import For, ParticleFor, While


class Visitor:
    def __init__(self, ast, max_depth=0):
        self.ast = ast
        self.max_depth = 0

    def visit(ast_node):
        for c in ast_node.children():
            if isinstance(c, Array):
                self.visit_Array(c)
            elif isinstance(c, BinOp):
                self.visit_BinOp(c)
            elif isinstance(c, (For, ParticleFor, While)):
                self.visit_Loop(c)
            else:
                self.visit(c)

    def visit_Array(self, ast_node):
        return self.visit(ast_node)

    def visit_BinOp(self, ast_node):
        return self.visit(ast_node)

    def visit_Loop(self, ast_node):
        return self.visit(ast_node)

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
