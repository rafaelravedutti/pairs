from pairs.ir.visitor import Visitor


class FetchModulesReferences(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.module_stack = []
        self.writing = False

    def visit_Assign(self, ast_node):
        self.writing = True
        for c in ast_node.destinations():
            self.visit(c)

        self.writing = False
        for c in ast_node.sources():
            self.visit(c)

    def visit_Module(self, ast_node):
        self.module_stack.append(ast_node)
        self.visit_children(ast_node)
        self.module_stack.pop()

    def visit_Array(self, ast_node):
        for m in self.module_stack:
            m.add_array(ast_node, self.writing)
            if m.run_on_device:
                ast_node.device_flag = True

    def visit_Property(self, ast_node):
        for m in self.module_stack:
            m.add_property(ast_node, self.writing)
            if m.run_on_device:
                ast_node.device_flag = True

    def visit_Var(self, ast_node):
        for m in self.module_stack:
            m.add_variable(ast_node, self.writing)
            if m.run_on_device:
                ast_node.device_flag = True
