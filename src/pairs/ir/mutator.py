class Mutator:
    def __init__(self, ast=None, max_depth=0):
        self.ast = ast
        self.max_depth = 0

    def set_ast(self, ast):
        self.ast = ast

    def get_method(self, method_name):
        method = getattr(self, method_name, None)
        return method if callable(method) else None

    def mutate(self, ast_node=None):
        if ast_node is None:
            ast_node = self.ast

        method = self.get_method(f"mutate_{type(ast_node).__name__}")
        if method is not None:
            return method(ast_node)

        method_unknown = self.get_method("mutate_Unknown")
        if method_unknown is not None:
            return method_unknown(ast_node)

        return ast_node

    def mutate_ArrayAccess(self, ast_node):
        ast_node.array = self.mutate(ast_node.array)
        ast_node.indexes = [self.mutate(i) for i in ast_node.indexes]

        if ast_node.index is not None:
            ast_node.index = self.mutate(ast_node.index)

        return ast_node 

    def mutate_Assign(self, ast_node):
        ast_node.assignments = [(self.mutate(a[0]), self.mutate(a[1])) for a in ast_node.assignments]
        return ast_node

    def mutate_BinOp(self, ast_node):
        ast_node.lhs = self.mutate(ast_node.lhs)
        ast_node.rhs = self.mutate(ast_node.rhs)
        ast_node.expressions = {i: self.mutate(e) for i, e in ast_node.expressions.items()}
        return ast_node

    def mutate_Block(self, ast_node):
        ast_node.stmts = [self.mutate(s) for s in ast_node.stmts]
        return ast_node

    def mutate_Branch(self, ast_node):
        ast_node.cond = self.mutate(ast_node.cond)
        ast_node.block_if = self.mutate(ast_node.block_if)
        ast_node.block_else = None if ast_node.block_else is None else self.mutate(ast_node.block_else)
        return ast_node

    def mutate_Cast(self, ast_node):
        ast_node.expr = self.mutate(ast_node.expr)
        return ast_node

    def mutate_Decl(self, ast_node):
        ast_node.elem = self.mutate(ast_node.elem)
        return ast_node

    def mutate_Filter(self, ast_node):
        return self.mutate_Branch(ast_node)

    def mutate_For(self, ast_node):
        ast_node.iterator = self.mutate(ast_node.iterator)
        ast_node.block = self.mutate(ast_node.block)
        return ast_node

    def mutate_Kernel(self, ast_node):
        ast_node._block = self.mutate(ast_node._block)
        return ast_node

    def mutate_KernelLaunch(self, ast_node):
        ast_node._kernel = self.mutate(ast_node._kernel)
        ast_node._iterator = self.mutate(ast_node._iterator)
        ast_node._range_min = self.mutate(ast_node._range_min)
        ast_node._range_max = self.mutate(ast_node._range_max)
        return ast_node

    def mutate_ParticleFor(self, ast_node):
        return self.mutate_For(ast_node)

    def mutate_PropertyAccess(self, ast_node):
        ast_node.prop = self.mutate(ast_node.prop)
        ast_node.index = self.mutate(ast_node.index)
        ast_node.expressions = {i: self.mutate(e) for i, e in ast_node.expressions.items()}
        return ast_node

    def mutate_Malloc(self, ast_node):
        ast_node.array = self.mutate(ast_node.array)
        ast_node.size = self.mutate(ast_node.size)
        return ast_node

    def mutate_Module(self, ast_node):
        ast_node._block = self.mutate(ast_node._block)
        return ast_node

    def mutate_ModuleCall(self, ast_node):
        ast_node._module = self.mutate(ast_node._module)
        return ast_node

    def mutate_Realloc(self, ast_node):
        ast_node.array = self.mutate(ast_node.array)
        ast_node.size = self.mutate(ast_node.size)
        return ast_node

    def mutate_Select(self, ast_node):
        ast_node.cond = self.mutate(ast_node.cond)
        ast_node.expr_if = self.mutate(ast_node.expr_if)
        ast_node.expr_else = self.mutate(ast_node.expr_else)
        return ast_node

    def mutate_Sqrt(self, ast_node):
        ast_node.expr = self.mutate(ast_node.expr)
        return ast_node

    def mutate_Timestep(self, ast_node):
        ast_node.block = self.mutate(ast_node.block)
        return ast_node

    def mutate_VectorAccess(self, ast_node):
        ast_node.expr = self.mutate(ast_node.expr)
        return ast_node

    def mutate_While(self, ast_node):
        ast_node.cond = self.mutate(ast_node.cond)
        ast_node.block = self.mutate(ast_node.block)
        return ast_node
