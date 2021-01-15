from ast.mutator import Mutator
from ast.visitor import Visitor


class SetBlockVariants(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        self.current_block = None
        self.in_assignment = None

    def mutate_Block(self, ast_node):
        self.current_block = ast_node
        ast_node.stmts = [self.mutate(s) for s in ast_node.stmts]
        return ast_node

    def mutate_Assign(self, ast_node):
        self.in_assignment = ast_node
        for dest in ast_node.destinations():
            self.mutate(dest)
        self.in_assignment = None
        return ast_node

    def mutate_Array(self, ast_node):
        if self.in_assignment is not None:
            self.in_assignment.parent_block.add_variant(ast_node)

        return ast_node

    def mutate_For(self, ast_node):
        ast_node.block.add_variant(ast_node.iterator)
        ast_node.iterator = self.mutate(ast_node.iterator)
        ast_node.block = self.mutate(ast_node.block)
        return ast_node

    def mutate_Property(self, ast_node):
        if self.in_assignment is not None:
            self.in_assignment.parent_block.add_variant(ast_node)

        return ast_node

    def mutate_Variable(self, ast_node):
        if self.in_assignment is not None:
            self.in_assignment.parent_block.add_variant(ast_node)

        return ast_node


class SetParentBlock(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.blocks = []

    def current_block(self):
        return self.blocks[-1]

    def visit_Assign(self, ast_node):
        ast_node.parent_block = self.current_block
        self.visit_children(ast_node)

    def visit_Block(self, ast_node):
        ast_node.parent_block = self.current_block
        self.blocks.append(ast_node)
        self.visit_children(ast_node)
        self.blocks.pop()

    def visit_BinOpDef(self, ast_node):
        ast_node.parent_block = self.current_block
        self.visit_children(ast_node)

    def visit_Branch(self, ast_node):
        ast_node.parent_block = self.current_block
        self.visit_children(ast_node)

    def visit_Filter(self, ast_node):
        ast_node.parent_block = self.current_block
        self.visit_children(ast_node)

    def visit_For(self, ast_node):
        ast_node.parent_block = self.current_block
        self.visit_children(ast_node)

    def visit_ParticleFor(self, ast_node):
        ast_node.parent_block = self.current_block
        self.visit_children(ast_node)

    def visit_Malloc(self, ast_node):
        ast_node.parent_block = self.current_block
        self.visit_children(ast_node)

    def visit_Realloc(self, ast_node):
        ast_node.parent_block = self.current_block
        self.visit_children(ast_node)

    def visit_While(self, ast_node):
        ast_node.parent_block = self.current_block
        self.visit_children(ast_node)

    def get_loop_parent_block(self, ast_node):
        assert isinstance(ast_node, (For, While)), "Node must be a loop!"
        loop_id = id(ast_node)
        return self.parents[loop_id] if loop_id in self.parents else None


class LICM(Mutator):
    def __init__(self, ast, loop_parents):
        super().__init__(ast)
        self.loop_parents = loop_parents

    def mutate_For(self, ast_node):
        ast_node.iterator = self.mutate(ast_node.iterator)
        ast_node.block = self.mutate(ast_node.block)
        return ast_node

    def mutate_Block(self, ast_node):
        ast_node.stmts = [self.mutate(s) for s in ast_node.stmts]
        return ast_node


def move_loop_invariant_code(ast):
    set_parent_block = SetParentBlock(ast)
    set_parent_block.visit()
    set_block_variants = SetBlockVariants(ast)
    set_block_variants.mutate()
    licm = LICM(ast)
    licm.mutate()
