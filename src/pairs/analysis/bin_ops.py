from pairs.ir.bin_op import BinOp
from pairs.ir.visitor import Visitor


class SetBinOpTerminals(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.elems = []

    def push_terminal(self, ast_node):
        for e in self.elems:
            e.add_terminal(ast_node.name())

    def visit_ArrayAccess(self, ast_node):
        self.elems.append(ast_node)
        self.visit_children(ast_node)
        self.elems.pop()

    def visit_BinOp(self, ast_node):
        self.elems.append(ast_node)
        self.visit_children(ast_node)
        self.elems.pop()

    def visit_PropertyAccess(self, ast_node):
        self.elems.append(ast_node)
        self.visit_children(ast_node)
        self.elems.pop()

    def visit_FeaturePropertyAccess(self, ast_node):
        self.elems.append(ast_node)
        self.visit_children(ast_node)
        self.elems.pop()

    def visit_Array(self, ast_node):
        self.push_terminal(ast_node)

    # TODO: Array should be enough
    def visit_ArrayND(self, ast_node):
        self.push_terminal(ast_node)

    def visit_Iter(self, ast_node):
        self.push_terminal(ast_node)

    def visit_Property(self, ast_node):
        self.push_terminal(ast_node)

    def visit_Var(self, ast_node):
        self.push_terminal(ast_node)


class ResetInPlaceBinOps(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)

    def visit_BinOp(self, ast_node):
        ast_node.in_place = True
        self.visit_children(ast_node)


class SetInPlaceBinOps(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)

    def visit_Decl(self, ast_node):
        if isinstance(ast_node.elem, BinOp):
            ast_node.elem.in_place = False


class SetDeclaredExprs(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.declared_exprs = []

    def visit_Decl(self, ast_node):
        self.declared_exprs.append(id(ast_node.elem))
