from pairs.ir.module import Module
from pairs.ir.mutator import Mutator
from pairs.ir.variables import Deref
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
            m.add_array(ast_node)

    def visit_Property(self, ast_node):
        for m in self.module_stack:
            m.add_property(ast_node)

    def visit_Var(self, ast_node):
        for m in self.module_stack:
            m.add_variable(ast_node, self.writing)


class AddDereferencesToWriteVariables(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        self.module_stack = []

    def mutate_Module(self, ast_node):
        self.module_stack.append(ast_node)
        ast_node._block = self.mutate(ast_node._block)
        self.module_stack.pop()
        return ast_node

    def mutate_Var(self, ast_node):
        parent_module = self.module_stack[-1]
        if parent_module.name != 'main' and ast_node in parent_module.write_variables():
            return Deref(ast_node.sim, ast_node)

        return ast_node


def fetch_modules_references(ast):
    fetch_refs = FetchModulesReferences(ast)
    fetch_refs.visit()
    add_derefs_to_write_vars = AddDereferencesToWriteVariables(ast)
    add_derefs_to_write_vars.mutate()
