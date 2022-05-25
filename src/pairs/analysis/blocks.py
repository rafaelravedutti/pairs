from pairs.ir.loops import For, While
from pairs.ir.mutator import Mutator
from pairs.ir.visitor import Visitor


class SetBlockVariants(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        self.in_assignment = None
        self.blocks = []

    def push_variant(self, ast_node):
        if self.in_assignment is not None:
            for block in self.blocks:
                block.add_variant(ast_node.name())

        return ast_node

    def mutate_Block(self, ast_node):
        self.blocks.append(ast_node)
        ast_node.stmts = [self.mutate(s) for s in ast_node.stmts]
        self.blocks.pop()
        return ast_node

    def mutate_Assign(self, ast_node):
        self.in_assignment = ast_node if ast_node.parent_block is not None else None
        for dest in ast_node.destinations():
            self.mutate(dest)
        self.in_assignment = None
        return ast_node

    def mutate_For(self, ast_node):
        self.push_variant(ast_node.iterator)
        ast_node.block.add_variant(ast_node.iterator.name())
        ast_node.iterator = self.mutate(ast_node.iterator)
        ast_node.block = self.mutate(ast_node.block)
        return ast_node

    def mutate_BinOp(self, ast_node):
        ast_node.lhs = self.mutate(ast_node.lhs)
        ast_node.rhs = self.mutate(ast_node.rhs)
        return ast_node

    def mutate_ArrayAccess(self, ast_node):
        # For array accesses, we only want to include the array name, and not
        # the index that is also present in the access node
        ast_node.array = self.mutate(ast_node.array)
        return ast_node

    def mutate_Array(self, ast_node):
        return self.push_variant(ast_node)

    # TODO: Array should be enough
    def mutate_ArrayND(self, ast_node):
        return self.push_variant(ast_node)

    def mutate_Iter(self, ast_node):
        return self.push_variant(ast_node)

    def mutate_Property(self, ast_node):
        return self.push_variant(ast_node)

    def mutate_PropertyAccess(self, ast_node):
        # For property accesses, we only want to include the property name, and not
        # the index that is also present in the access node
        ast_node.prop = self.mutate(ast_node.prop)
        return ast_node

    def mutate_Var(self, ast_node):
        return self.push_variant(ast_node)


class SetParentBlock(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.blocks = []

    def current_block(self):
        return self.blocks[-1] if self.blocks else None

    def set_parent_block(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.visit_children(ast_node)

    def visit_Block(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.blocks.append(ast_node)
        self.visit_children(ast_node)
        self.blocks.pop()

    def visit_Assign(self, ast_node):
        self.set_parent_block(ast_node)

    def visit_Branch(self, ast_node):
        self.set_parent_block(ast_node)

    def visit_Decl(self, ast_node):
        self.set_parent_block(ast_node)

    def visit_Filter(self, ast_node):
        self.set_parent_block(ast_node)

    def visit_For(self, ast_node):
        self.set_parent_block(ast_node)

    def visit_ParticleFor(self, ast_node):
        self.set_parent_block(ast_node)

    def visit_Malloc(self, ast_node):
        self.set_parent_block(ast_node)

    def visit_Realloc(self, ast_node):
        self.set_parent_block(ast_node)

    def visit_While(self, ast_node):
        self.set_parent_block(ast_node)

    def get_loop_parent_block(self, ast_node):
        assert isinstance(ast_node, (For, While)), "Node must be a loop!"
        loop_id = id(ast_node)
        return self.parents[loop_id] if loop_id in self.parents else None


class SetExprOwnerBlock(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.ownership = {}
        self.expressions_to_lift = []
        self.block_level = {}
        self.block_parent = {}
        self.block_stack = []

    def common_parent_block(self, block1, block2):
        if block1 is None:
            return (block2, False)

        if block2 is None:
            return (block1, False)

        parent_block1 = block1
        parent_block2 = block2
        while parent_block1 != parent_block2:
            l1 = self.block_level[parent_block1]
            l2 = self.block_level[parent_block2]

            if l1 >= l2:
                if l1 == 0:
                    return (parent_block1, False)

                parent_block1 = self.block_parent[parent_block1]

            if l2 >= l1:
                if l2 == 0:
                    return (parent_block2, False)

                parent_block2 = self.block_parent[parent_block2]

        return (parent_block1, parent_block1 != block1 and parent_block1 != block2)

    def set_ownership(self, ast_node):
        if ast_node not in self.ownership:
            self.ownership[ast_node] = None

        self.ownership[ast_node], must_lift = self.common_parent_block(self.ownership[ast_node], self.block_stack[-1])
        if must_lift and ast_node not in self.expressions_to_lift:
            self.expressions_to_lift.append(ast_node)

    def visit_Block(self, ast_node):
        self.block_level[ast_node] = len(self.block_stack)
        self.block_parent[ast_node] = self.block_stack[-1] if len(self.block_stack) > 0 else None
        self.block_stack.append(ast_node)
        self.visit_children(ast_node)
        self.block_stack.pop()

    def visit_BinOp(self, ast_node):
        self.set_ownership(ast_node)
        self.visit_children(ast_node)

    def visit_PropertyAccess(self, ast_node):
        self.set_ownership(ast_node)
        self.visit_children(ast_node)
