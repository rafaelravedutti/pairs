import pairs.ir.utils as util


class Mutator:
    def __init__(self, ast=None, max_depth=0):
        self.ast = ast
        self.max_depth = 0
        self.visited_nodes = []

    def set_ast(self, ast):
        self.ast = ast

    def get_method(self, method_name):
        method = getattr(self, method_name, None)
        return method if callable(method) else None

    def mutate(self, ast_node=None):
        if ast_node is None:
            ast_node = self.ast

        terminal_node = util.is_terminal(ast_node)
        if terminal_node or ast_node not in self.visited_nodes:
            if not terminal_node:
                self.visited_nodes.append(ast_node)

            method = self.get_method(f"mutate_{type(ast_node).__name__}")
            if method is not None:
                return method(ast_node)

            for b in type(ast_node).__bases__:
                method = self.get_method(f"mutate_{b.__name__}")
                if method is not None:
                    return method(ast_node)

            method_unknown = self.get_method("mutate_Unknown")
            if method_unknown is not None:
                return method_unknown(ast_node)

        return ast_node

    def mutate_ArrayAccess(self, ast_node):
        ast_node.array = self.mutate(ast_node.array)
        ast_node.partial_indexes = [self.mutate(i) for i in ast_node.partial_indexes]

        if ast_node.flat_index is not None:
            ast_node.flat_index = self.mutate(ast_node.flat_index)

        return ast_node 

    def mutate_Assign(self, ast_node):
        ast_node._dest = self.mutate(ast_node._dest)
        ast_node._src = self.mutate(ast_node._src)
        return ast_node

    def mutate_AtomicAdd(self, ast_node):
        ast_node.elem = self.mutate(ast_node.elem)
        ast_node.value = self.mutate(ast_node.value)

        if ast_node.check_for_resize():
            ast_node.resize = self.mutate(ast_node.resize)
            ast_node.capacity = self.mutate(ast_node.capacity)

        return ast_node

    def mutate_Block(self, ast_node):
        ast_node.stmts = [self.mutate(s) for s in ast_node.stmts]
        return ast_node

    def mutate_Branch(self, ast_node):
        ast_node.cond = self.mutate(ast_node.cond)
        ast_node.block_if = self.mutate(ast_node.block_if)
        ast_node.block_else = None if ast_node.block_else is None else self.mutate(ast_node.block_else)
        return ast_node

    def mutate_Call(self, ast_node):
        ast_node.params = [self.mutate(p) for p in ast_node.params]
        return ast_node

    def mutate_Call_Void(self, ast_node):
        ast_node.params = [self.mutate(p) for p in ast_node.params]
        return ast_node

    def mutate_Cast(self, ast_node):
        ast_node.expr = self.mutate(ast_node.expr)
        return ast_node

    def mutate_Decl(self, ast_node):
        ast_node.elem = self.mutate(ast_node.elem)
        return ast_node

    def mutate_DeviceStaticRef(self, ast_node):
        ast_node.elem = self.mutate(ast_node.elem)
        return ast_node

    def mutate_Filter(self, ast_node):
        return self.mutate_Branch(ast_node)

    def mutate_For(self, ast_node):
        ast_node.iterator = self.mutate(ast_node.iterator)
        ast_node.block = self.mutate(ast_node.block)
        ast_node.min = self.mutate(ast_node.min)
        ast_node.max = self.mutate(ast_node.max)
        return ast_node

    def mutate_HostRef(self, ast_node):
        ast_node.elem = self.mutate(ast_node.elem)
        return ast_node

    def mutate_Kernel(self, ast_node):
        ast_node._block = self.mutate(ast_node._block)
        return ast_node

    def mutate_KernelLaunch(self, ast_node):
        ast_node._kernel = self.mutate(ast_node._kernel)
        ast_node._iterator = self.mutate(ast_node._iterator)
        ast_node._range_min = self.mutate(ast_node._range_min)
        ast_node._range_max = self.mutate(ast_node._range_max)
        ast_node._threads_per_block = self.mutate(ast_node._threads_per_block)
        ast_node._nblocks = self.mutate(ast_node._nblocks)
        return ast_node

    def mutate_ParticleFor(self, ast_node):
        return self.mutate_For(ast_node)

    def mutate_PropertyAccess(self, ast_node):
        ast_node.prop = self.mutate(ast_node.prop)
        ast_node.index = self.mutate(ast_node.index)
        ast_node.vector_indexes = {d: self.mutate(i) for d, i in ast_node.vector_indexes.items()}
        return ast_node

    def mutate_ContactPropertyAccess(self, ast_node):
        ast_node.contact_prop = self.mutate(ast_node.contact_prop)
        ast_node.index = self.mutate(ast_node.index)
        ast_node.vector_indexes = {d: self.mutate(i) for d, i in ast_node.vector_indexes.items()}
        return ast_node

    def mutate_FeaturePropertyAccess(self, ast_node):
        ast_node.feature_prop = self.mutate(ast_node.feature_prop)
        ast_node.index = self.mutate(ast_node.index)
        ast_node.vector_indexes = {d: self.mutate(i) for d, i in ast_node.vector_indexes.items()}
        return ast_node

    def mutate_Malloc(self, ast_node):
        ast_node.array = self.mutate(ast_node.array)
        ast_node.size = self.mutate(ast_node.size)
        return ast_node

    def mutate_MathFunction(self, ast_node):
        ast_node._params = [self.mutate(p) for p in ast_node._params]
        return ast_node

    def mutate_Module(self, ast_node):
        ast_node._block = self.mutate(ast_node._block)
        return ast_node

    def mutate_ModuleCall(self, ast_node):
        ast_node._module = self.mutate(ast_node._module)
        return ast_node

    def mutate_Neighbor(self, ast_node):
        ast_node._neighbor_index = self.mutate(ast_node._neighbor_index)
        ast_node._particle_index = self.mutate(ast_node._particle_index)

        if ast_node._cell_id is not None:
            ast_node._cell_id = self.mutate(ast_node._cell_id)

        return ast_node

    def mutate_Realloc(self, ast_node):
        ast_node.array = self.mutate(ast_node.array)
        ast_node.size = self.mutate(ast_node.size)
        return ast_node

    def mutate_ScalarOp(self, ast_node):
        ast_node.lhs = self.mutate(ast_node.lhs)

        if not ast_node.operator().is_unary():
            ast_node.rhs = self.mutate(ast_node.rhs)

        return ast_node

    def mutate_Select(self, ast_node):
        ast_node.cond = self.mutate(ast_node.cond)
        ast_node.expr_if = self.mutate(ast_node.expr_if)
        ast_node.expr_else = self.mutate(ast_node.expr_else)
        return ast_node

    def mutate_Timestep(self, ast_node):
        ast_node.block = self.mutate(ast_node.block)
        return ast_node

    def mutate_Vector(self, ast_node):
        ast_node._values = [self.mutate(v) for v in ast_node._values]
        return ast_node

    def mutate_VectorAccess(self, ast_node):
        ast_node.expr = self.mutate(ast_node.expr)
        return ast_node

    def mutate_VectorOp(self, ast_node):
        ast_node.lhs = self.mutate(ast_node.lhs)

        if not ast_node.operator().is_unary():
            ast_node.rhs = self.mutate(ast_node.rhs)

        return ast_node

    def mutate_While(self, ast_node):
        ast_node.cond = self.mutate(ast_node.cond)
        ast_node.block = self.mutate(ast_node.block)
        return ast_node
