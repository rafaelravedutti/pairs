from pairs.ir.arrays import ArrayAccess
from pairs.ir.branches import Branch
from pairs.ir.lit import Lit
from pairs.ir.loops import For
from pairs.ir.quaternions import QuaternionOp
from pairs.ir.scalars import ScalarOp
from pairs.ir.matrices import MatrixOp
from pairs.ir.visitor import Visitor
from pairs.ir.vectors import VectorOp


class MarkCandidateLoops(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)

    def visit_Module(self, ast_node):
        possible_candidates = []
        for stmt in ast_node._block.stmts:
            if stmt is not None:
                if isinstance(stmt, Branch):
                    for branch_stmt in stmt.block_if.stmts:
                        if isinstance(branch_stmt, For):
                            possible_candidates.append(branch_stmt)

                    if stmt.block_else is not None:
                        for branch_stmt in stmt.block_else.stmts:
                            if isinstance(branch_stmt, For):
                                possible_candidates.append(branch_stmt)

                if isinstance(stmt, For):
                    possible_candidates.append(stmt)

        for stmt in possible_candidates:
            if not isinstance(stmt.min, Lit) or not isinstance(stmt.max, Lit):
                stmt.mark_as_kernel_candidate()

        self.visit_children(ast_node)


class FetchKernelReferences(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.kernel_stack = []
        self.kernel_decls = {}
        self.kernel_used_array_accesses = {}
        self.kernel_used_scalar_ops = {}
        self.kernel_used_vector_ops = {}
        self.kernel_used_matrix_ops = {}
        self.kernel_used_quat_ops = {}
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
        self.writing = False
        self.visit(ast_node._src)
        self.writing = True
        self.visit(ast_node._dest)
        self.writing = False

    def visit_AtomicAdd(self, ast_node):
        self.writing = True
        self.visit(ast_node.elem)
        self.writing = False
        self.visit(ast_node.value)

        if ast_node.resize is not None:
            self.visit(ast_node.resize)
            self.visit(ast_node.capacity)

    def visit_AtomicInc(self, ast_node):
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
        self.kernel_used_matrix_ops[kernel_id] = []
        self.kernel_used_quat_ops[kernel_id] = []
        self.kernel_stack.append(ast_node)
        self.visit_children(ast_node)
        self.kernel_stack.pop()

        ast_node.add_array_access(
            [a for a in self.kernel_used_array_accesses[kernel_id] \
             if a not in self.kernel_decls[kernel_id]])

        ast_node.add_scalar_op(
            [b for b in self.kernel_used_scalar_ops[kernel_id] \
            if b not in self.kernel_decls[kernel_id] and not b.in_place])

        ast_node.add_vector_op(
            [b for b in self.kernel_used_vector_ops[kernel_id] \
            if b not in self.kernel_decls[kernel_id] and not b.in_place])

        ast_node.add_matrix_op(
            [b for b in self.kernel_used_matrix_ops[kernel_id] \
            if b not in self.kernel_decls[kernel_id] and not b.in_place])

        ast_node.add_quaternion_op(
            [b for b in self.kernel_used_quat_ops[kernel_id] \
            if b not in self.kernel_decls[kernel_id] and not b.in_place])

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
        if isinstance(ast_node.elem, (ArrayAccess, ScalarOp, VectorOp, MatrixOp, QuaternionOp)):
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

    def visit_MatrixOp(self, ast_node):
        for k in self.kernel_stack:
            self.kernel_used_matrix_ops[k.kernel_id].append(ast_node)

        self.visit_children(ast_node)

    def visit_QuaternionOp(self, ast_node):
        for k in self.kernel_stack:
            self.kernel_used_quat_ops[k.kernel_id].append(ast_node)

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
