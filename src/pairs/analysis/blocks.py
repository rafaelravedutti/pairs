from pairs.ir.loops import For, While
from pairs.ir.mutator import Mutator
from pairs.ir.visitor import Visitor


class SetBlockVariants(Mutator):
    def __init__(self, ast=None):
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
        self.in_assignment = ast_node if len(self.blocks) > 0 else None
        for dest in ast_node.destinations():
            self.mutate(dest)
        self.in_assignment = None
        return ast_node

    def mutate_AtomicAdd(self, ast_node):
        self.in_assignment = ast_node
        ast_node.elem = self.mutate(ast_node.elem)
        self.in_assignment = None
        ast_node.value = self.mutate(ast_node.value)

        if ast_node.check_for_resize():
            ast_node.resize = self.mutate(ast_node.resize)
            ast_node.capacity = self.mutate(ast_node.capacity)

        return ast_node

    def mutate_For(self, ast_node):
        self.push_variant(ast_node.iterator)
        ast_node.block.add_variant(ast_node.iterator.name())
        ast_node.iterator = self.mutate(ast_node.iterator)
        ast_node.block = self.mutate(ast_node.block)
        return ast_node

    def mutate_BinOp(self, ast_node):
        ast_node.lhs = self.mutate(ast_node.lhs)
        if not ast_node.operator().is_unary():
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

    def mutate_ContactProperty(self, ast_node):
        return self.push_variant(ast_node)

    def mutate_FeatureProperty(self, ast_node):
        return self.push_variant(ast_node)

    def mutate_PropertyAccess(self, ast_node):
        # For property accesses, we only want to include the property name, and not
        # the index that is also present in the access node
        ast_node.prop = self.mutate(ast_node.prop)
        return ast_node

    def mutate_ContactPropertyAccess(self, ast_node):
        # For property accesses, we only want to include the property name, and not
        # the index that is also present in the access node
        ast_node.contact_prop = self.mutate(ast_node.contact_prop)
        return ast_node

    def mutate_FeaturePropertyAccess(self, ast_node):
        # For property accesses, we only want to include the property name, and not
        # the index that is also present in the access node
        ast_node.feature_prop = self.mutate(ast_node.feature_prop)
        return ast_node

    def mutate_Var(self, ast_node):
        return self.push_variant(ast_node)


class SetParentBlock(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.blocks = []
        self.block_statement = {}

    def visit_Block(self, ast_node):
        parent_block = self.blocks[-1] if self.blocks else None
        ast_node.parent_block = parent_block
        ast_node.parent_statement = self.block_statement[parent_block]

        self.blocks.append(ast_node)
        for s in ast_node.statements():
            s.parent_block = ast_node
            s.parent_statement = None # Statements have no parent statement
            self.block_statement[ast_node] = s
            self.visit_children(s)

        self.blocks.pop()


class DetermineExpressionsOwnership(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.ownership = {}
        self.expressions_to_lift = []
        self.block_location = {}
        self.block_level = {}
        self.block_statement = {}
        self.block_stack = []

    def find_common_ownership(self, location1, location2):
        block1, stmt1 = location1
        block2, stmt2 = location2

        while block1 != block2:
            level1 = self.block_level[block1]
            level2 = self.block_level[block2]

            if level1 >= level2:
                block1, stmt1 = self.block_location[block1]

            if level2 >= level1:
                block2, stmt2 = self.block_location[block2]

        for s in block1.statements():
            if s == stmt1 or s == stmt2:
                return (block1, s)

        return (block1, stmt1)

    def location(self):
        if len(self.block_stack) <= 0:
            return (None, None)

        parent_block = self.block_stack[-1]
        return (parent_block, self.block_statement[parent_block])

    def update_ownership(self, ast_node):
        if ast_node not in self.ownership:
            self.ownership[ast_node] = self.location()

        else:
            common_ownership = self.find_common_ownership(self.ownership[ast_node], self.location())
            common_block, common_stmt = common_ownership
            owner_block, owner_stmt = self.ownership[ast_node]

            if self.block_level[common_block] < self.block_level[owner_block]:
                self.ownership[ast_node] = common_ownership

                if ast_node not in self.expressions_to_lift:
                    self.expressions_to_lift.append(ast_node)

    def visit_Block(self, ast_node):
        self.block_location[ast_node] = self.location()
        self.block_level[ast_node] = len(self.block_stack)
        self.block_stack.append(ast_node)
        for s in ast_node.statements():
            self.block_statement[ast_node] = s
            self.visit_children(s)

        self.block_stack.pop()

    def visit_BinOp(self, ast_node):
        self.visit_children(ast_node)
        self.update_ownership(ast_node)

    def visit_MathFunction(self, ast_node):
        self.visit_children(ast_node)
        self.update_ownership(ast_node)

    def visit_PropertyAccess(self, ast_node):
        self.visit_children(ast_node)
        self.update_ownership(ast_node)

    def visit_ContactPropertyAccess(self, ast_node):
        self.visit_children(ast_node)
        self.update_ownership(ast_node)

    def visit_FeaturePropertyAccess(self, ast_node):
        self.visit_children(ast_node)
        self.update_ownership(ast_node)

    def visit_Select(self, ast_node):
        self.visit_children(ast_node)
        self.update_ownership(ast_node)
