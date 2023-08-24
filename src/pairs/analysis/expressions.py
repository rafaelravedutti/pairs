from pairs.ir.scalars import ScalarOp
from pairs.ir.vectors import VectorOp
from pairs.ir.visitor import Visitor


class DetermineExpressionsTerminals(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.elems = []

    def traverse_expression(self, ast_node):
        self.elems.append(ast_node)
        self.clear_visited_nodes()
        self.visit_children(ast_node)
        self.elems.pop()

    def push_terminal(self, ast_node):
        for e in self.elems:
            e.add_terminal(ast_node.name())

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
