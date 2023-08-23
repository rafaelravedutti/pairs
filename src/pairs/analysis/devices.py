from pairs.ir.arrays import ArrayAccess
from pairs.ir.scalars import ScalarOp
from pairs.ir.visitor import Visitor
from pairs.ir.vectors import VectorOp


class FetchKernelReferences(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.kernel_stack = []
        self.kernel_decls = {}
        self.kernel_used_array_accesses = {}
        self.kernel_used_scalar_ops = {}
        self.kernel_used_vector_ops = {}
        self.writing = False

    def visit_ArrayAccess(self, ast_node):
        if not self.writing and ast_node.inlined is False:
            for k in self.kernel_stack:
                self.kernel_used_array_accesses[k.kernel_id].append(ast_node)

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

        if ast_node.resize is not None:
            self.visit(ast_node.resize)
            self.visit(ast_node.capacity)

    def visit_Kernel(self, ast_node):
        kernel_id = ast_node.kernel_id
        self.kernel_decls[kernel_id] = []
        self.kernel_used_array_accesses[kernel_id] = []
        self.kernel_used_scalar_ops[kernel_id] = []
        self.kernel_used_vector_ops[kernel_id] = []
        self.kernel_stack.append(ast_node)
        self.visit_children(ast_node)
        self.kernel_stack.pop()
        ast_node.add_array_access([a for a in self.kernel_used_array_accesses[kernel_id] if a not in self.kernel_decls[kernel_id]])
        ast_node.add_scalar_op([b for b in self.kernel_used_scalar_ops[kernel_id] if b not in self.kernel_decls[kernel_id] and not b.in_place])
        ast_node.add_vector_op([b for b in self.kernel_used_vector_ops[kernel_id] if b not in self.kernel_decls[kernel_id] and not b.in_place])

    def visit_PropertyAccess(self, ast_node):
        # Visit property and save current writing state
        self.visit(ast_node.prop)
        writing_state = self.writing

        # Index elements are read-only
        self.writing = False
        self.visit([roc for roc in ast_node.children() if roc != ast_node.prop])
        self.writing = writing_state

    def visit_ContactPropertyAccess(self, ast_node):
        # Visit property and save current writing state
        self.visit(ast_node.contact_prop)
        writing_state = self.writing

        # Index elements are read-only
        self.writing = False
        self.visit([roc for roc in ast_node.children() if roc != ast_node.contact_prop])
        self.writing = writing_state

    def visit_FeaturePropertyAccess(self, ast_node):
        # Visit property and save current writing state
        self.visit(ast_node.feature_prop)
        writing_state = self.writing

        # Index elements are read-only
        self.writing = False
        self.visit([roc for roc in ast_node.children() if roc != ast_node.feature_prop])
        self.writing = writing_state

    def visit_Decl(self, ast_node):
        if isinstance(ast_node.elem, (ArrayAccess, ScalarOp, VectorOp)):
            for k in self.kernel_stack:
                self.kernel_decls[k.kernel_id].append(ast_node.elem)

    def visit_ScalarOp(self, ast_node):
        if ast_node.inlined is False:
            for k in self.kernel_stack:
                self.kernel_used_scalar_ops[k.kernel_id].append(ast_node)

        self.visit_children(ast_node)

    def visit_VectorOp(self, ast_node):
        for k in self.kernel_stack:
            self.kernel_used_vector_ops[k.kernel_id].append(ast_node)

        self.visit_children(ast_node)

    def visit_Array(self, ast_node):
        for k in self.kernel_stack:
            k.add_array(ast_node, self.writing)

    def visit_Property(self, ast_node):
        for k in self.kernel_stack:
            k.add_property(ast_node, self.writing)

    def visit_ContactProperty(self, ast_node):
        for k in self.kernel_stack:
            k.add_contact_property(ast_node)

    def visit_FeatureProperty(self, ast_node):
        for k in self.kernel_stack:
            k.add_feature_property(ast_node)

    def visit_Var(self, ast_node):
        for k in self.kernel_stack:
            k.add_variable(ast_node, self.writing)

            # Variables only have a device version when changed within kernels
            if self.writing:
                ast_node.device_flag = True
