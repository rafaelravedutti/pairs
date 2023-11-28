from pairs.ir.visitor import Visitor


class FetchModulesReferences(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.module_stack = []
        self.writing = False

    def visit_ArrayAccess(self, ast_node):
        # Visit array and save current writing state
        self.visit(ast_node.array)
        writing_state = self.writing

        # Index elements are read-only
        self.writing = False
        self.visit([roc for roc in ast_node.children() if roc != ast_node.array])
        self.writing = writing_state

    def visit_Assign(self, ast_node):
        self.writing = True
        self.visit(ast_node._dest)
        self.writing = False
        self.visit(ast_node._src)

    def visit_AtomicAdd(self, ast_node):
        self.writing = True
        self.visit(ast_node.elem)
        self.writing = False
        self.visit(ast_node.value)

        for m in self.module_stack:
            if m.run_on_device:
                ast_node.device_flag = True

        if ast_node.resize is not None:
            self.visit(ast_node.resize)
            self.visit(ast_node.capacity)

    def visit_Module(self, ast_node):
        self.module_stack.append(ast_node)
        self.visit_children(ast_node)
        self.module_stack.pop()

    def visit_PropertyAccess(self, ast_node):
        # Visit property and save current writing state
        self.visit(ast_node.prop)
        writing_state = self.writing

        # Index elements are read-only
        self.writing = False
        self.visit([roc for roc in ast_node.children() if roc != ast_node.prop])
        self.writing = writing_state

    def visit_FeaturePropertyAccess(self, ast_node):
        # Visit property and save current writing state
        self.visit(ast_node.feature_prop)
        writing_state = self.writing

        # Index elements are read-only
        self.writing = False
        self.visit([roc for roc in ast_node.children() if roc != ast_node.feature_prop])
        self.writing = writing_state

    def visit_ContactPropertyAccess(self, ast_node):
        # Visit property and save current writing state
        self.visit(ast_node.contact_prop)
        writing_state = self.writing

        # Index elements are read-only
        self.writing = False
        self.visit([roc for roc in ast_node.children() if roc != ast_node.contact_prop])
        self.writing = writing_state

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

    def visit_ContactProperty(self, ast_node):
        for m in self.module_stack:
            m.add_contact_property(ast_node)
            if m.run_on_device:
                ast_node.device_flag = True

    def visit_FeatureProperty(self, ast_node):
        for m in self.module_stack:
            m.add_feature_property(ast_node)
            if m.run_on_device:
                ast_node.device_flag = True

    def visit_Var(self, ast_node):
        for m in self.module_stack:
            if not ast_node.temporary():
                m.add_variable(ast_node, self.writing)
