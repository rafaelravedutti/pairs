from pairs.ir.scalars import ScalarOp
from pairs.ir.vectors import VectorOp
from pairs.ir.visitor import Visitor


class DetermineExpressionsTerminals(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast, visit_nodes_once=False)
        self.expressions_stack = []
        self.visited_expressions = []

    def traverse_expression(self, ast_node):
        if ast_node in self.visited_expressions:
            for term in ast_node.terminals:
                for expr in self.expressions_stack:
                    expr.add_terminal(term)

        else:
            self.expressions_stack.append(ast_node)
            self.visit_children(ast_node)
            self.expressions_stack.pop()
            self.visited_expressions.append(ast_node)

    def push_terminal(self, ast_node):
        for expr in self.expressions_stack:
            expr.add_terminal(ast_node.name())

    def visit_ArrayAccess(self, ast_node):
        self.traverse_expression(ast_node)

    def visit_MathFunction(self, ast_node):
        self.traverse_expression(ast_node)

    def visit_PropertyAccess(self, ast_node):
        self.traverse_expression(ast_node)

    def visit_ContactPropertyAccess(self, ast_node):
        self.traverse_expression(ast_node)

    def visit_FeaturePropertyAccess(self, ast_node):
        self.traverse_expression(ast_node)

    def visit_ScalarOp(self, ast_node):
        self.traverse_expression(ast_node)

    def visit_Select(self, ast_node):
        self.traverse_expression(ast_node)

    def visit_Vector(self, ast_node):
        self.traverse_expression(ast_node)

    def visit_VectorOp(self, ast_node):
        self.traverse_expression(ast_node)

    def visit_Array(self, ast_node):
        self.push_terminal(ast_node)

    # TODO: Array should be enough
    def visit_ArrayND(self, ast_node):
        self.push_terminal(ast_node)

    def visit_Iter(self, ast_node):
        self.push_terminal(ast_node)

    def visit_Property(self, ast_node):
        self.push_terminal(ast_node)

    def visit_ContactProperty(self, ast_node):
        self.push_terminal(ast_node)

    def visit_FeatureProperty(self, ast_node):
        self.push_terminal(ast_node)

    def visit_Var(self, ast_node):
        self.push_terminal(ast_node)


class ResetInPlaceOperations(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)

    def visit_ScalarOp(self, ast_node):
        ast_node.in_place = True
        self.visit_children(ast_node)

    def visit_VectorOp(self, ast_node):
        ast_node.in_place = True
        self.visit_children(ast_node)


class DetermineInPlaceOperations(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)

    def visit_Decl(self, ast_node):
        if isinstance(ast_node.elem, (ScalarOp, VectorOp)):
            ast_node.elem.in_place = False


class ListDeclaredExpressions(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.declared_exprs = []

    def visit_Decl(self, ast_node):
        self.declared_exprs.append(id(ast_node.elem))
