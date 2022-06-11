from pairs.ir.bin_op import BinOp
from pairs.ir.visitor import Visitor


class SetBinOpTerminals(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.elems = []

    def push_terminal(self, ast_node):
        for e in self.elems:
            e.add_terminal(ast_node.name())

    def visit_BinOp(self, ast_node):
        self.elems.append(ast_node)
        self.visit_children(ast_node)
        self.elems.pop()

    def visit_PropertyAccess(self, ast_node):
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


class SetUsedBinOps(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.bin_ops = []
        self.writing = False

    def visit_Assign(self, ast_node):
        self.writing = True
        self.visit(ast_node.destinations())
        self.writing = False
        self.visit(ast_node.sources())

    def visit_BinOp(self, ast_node):
        ast_node.decl.used = True
        self.visit_children(ast_node)

    def visit_Decl(self, ast_node):
        pass

    def visit_PropertyAccess(self, ast_node):
        ast_node.decl.used = not self.writing
        self.writing = False
        self.visit_children(ast_node)


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
