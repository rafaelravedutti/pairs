from pairs.ir.bin_op import BinOp
from pairs.ir.branches import Branch
from pairs.ir.module import Module, Module_Call
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


class AddResizeLogic(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        self.block_stack = []
        self.module_stack = []
        self.module_resizes = {}
        self.resizes_to_check = {}
        self.check_properties_resize = False
        self.match_capacity = None
        self.update = {}
        self.resize_buffers = {}
        self.nresize_buffers = 0

    def mutate_Array(self, ast_node):
        for capacity, size in self.resizes_to_check.items():
            if size == ast_node.name():
                self.match_capacity = capacity

        return ast_node

    def mutate_Assignment(self, ast_node):
        for dest, src in ast_node.assignments.items():
            if isinstance(dest, ArrayAccess):
                self.match_capacity = None
                ast_node.indexes = [self.mutate(i) for i in ast_node.indexes]
                if ast_node.index is not None:
                    ast_node.index = self.mutate(ast_node.index)

                # Resize var is used in index, this statement should be checked for safety
                if self.match_capacity is not None:
                    size = self.resizes_to_check[match_capacity]
                    check_value = self.update[size] if size in self.update else size
                    resize_id = self.resize_buffers[match_capacity]
                    return Branch(ast_node.sim, check_value < match_capacity,
                                  Block(ast_node.sim, ast_node),
                                  Block(ast_node.sim, ast_node.resizes[resize_id].set(check_value)))

                # Size is changed here, assigned value must be used for further checkings
                for capacity, size in self.resizes_to_check.items():
                    if size == dest.array.name():
                        self.update[size] = src

            if isinstance(dest, Var):
                # Size is changed here, assigned value must be used for further checkings
                for capacity, size in self.resizes_to_check.items():
                    if size == dest.name():
                        self.update[size] = src

        return ast_node

    def mutate_Block(self, ast_node):
        self.block_stack.append(ast_node)
        ast_node.stmts = [self.mutate(s) for s in ast_node.stmts]
        self.block_stack.pop()
        return ast_node

    def mutate_Module(self, ast_node):
        # Save current state
        saved_resizes_to_check = self.resizes_to_check
        saved_check_properties_resize = self.check_properties_resize
        saved_update = self.update
        saved_resize_buffers = self.resize_buffers
        saved_nresize_buffers = self.nresize_buffers

        # Update state and keep traversing tree
        self.module_resizes[ast_node] = []
        self.module_stack.append(ast_node)
        for capacity in ast_node._resizes_to_check.keys():
            self.module_resizes[ast_node].append(self.nresize_buffers)
            self.resize_buffers[capacity] = self.nresize_buffers
            self.nresize_buffers += 1

        self.resizes_to_check = ast_node._resizes_to_check
        self.check_properties_resize = ast_node._check_properties_resize
        self.update = {}
        ast_node._block = self.mutate(ast_node._block)
        self.module_stack.pop()

        # Restore saved state
        self.resizes_to_check = saved_resizes_to_check
        self.check_properties_resize = saved_check_properties_resize
        self.update = saved_update
        self.resize_buffers = saved_resize_buffers
        self.nresize_buffers = saved_nresize_buffers
        return ast_node

    def mutate_Var(self, ast_node):
        for capacity, size in self.resizes_to_check.items():
            if size == ast_node.name():
                self.match_capacity = capacity

        return ast_node


class ReplaceModulesByCalls(Mutator):
    def __init__(self, ast, module_resizes):
        super().__init__(ast)
        self.module_resizes = module_resizes

    def mutate_Module(self, ast_node):
        ast_node._block = self.mutate(ast_node._block)
        if ast_node.name == 'main':
            return ast_node

        call = Module_Call(ast_node.sim, ast_node)
        if self.module_resizes[ast_node]:
            init_stmts = []
            reset_stmts = []
            branch_cond = None

            for r in self.module_resizes[ast_node]:
                init_stmts.append(Assign(ast_node.resizes[r], 1))
                reset_stmts.append(Assign(ast_node.resizes[r], 0))
                cond = ast_node.resizes[r] > 0
                branch_cond = cond if branch_cond is None else BinOp.or_op(cond, branch_cond)

            return Block(ast_node.sim, init_stmts + Filter(ast_node.sim, branch_cond, reset_stmts + [call]))

        return call


def modularize(ast):
    add_resize_logic = AddResizeLogic(ast)
    add_resize_logic.mutate()
    fetch_refs = FetchModulesReferences(ast)
    fetch_refs.visit()
    add_derefs_to_write_vars = AddDereferencesToWriteVariables(ast)
    add_derefs_to_write_vars.mutate()
    replace = ReplaceModulesByCalls(ast, add_resize_logic.module_resizes)
    replace.mutate()
