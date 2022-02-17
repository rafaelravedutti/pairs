from pairs.ir.visitor import Visitor


class SetBinOpTerminals(Visitor):
    def __init__(self, ast):
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
    def __init__(self, ast):
        super().__init__(ast)
        self.bin_ops = []

    def visit_BinOp(self, ast_node):
        ast_node.decl.used = True
        self.visit_children(ast_node)

    def visit_Decl(self, ast_node):
        pass

    def visit_PropertyAccess(self, ast_node):
        ast_node.decl.used = True
        self.visit_children(ast_node)
