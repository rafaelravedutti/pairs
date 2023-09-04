from pairs.ir.loops import For, While
from pairs.ir.visitor import Visitor


class DiscoverBlockVariants(Visitor):
    def __init__(self, ast=None):
        super().__init__(ast, visit_nodes_once=False)
        self.in_assignment = None
        self.blocks = []

    def push_variant(self, ast_node):
        if self.in_assignment is not None:
            for block in self.blocks:
                block.add_variant(ast_node.name())

    def visit_Block(self, ast_node):
        self.blocks.append(ast_node)
        self.visit_children(ast_node)
        self.blocks.pop()

    def visit_Assign(self, ast_node):
        self.in_assignment = ast_node if len(self.blocks) > 0 else None
        self.visit(ast_node._dest)
        self.in_assignment = None

    def visit_AtomicAdd(self, ast_node):
        self.in_assignment = ast_node
        self.visit(ast_node.elem)
        self.in_assignment = None
        self.visit(ast_node.value)

        if ast_node.check_for_resize():
            self.visit(ast_node.resize)
            self.visit(ast_node.capacity)

    def visit_For(self, ast_node):
        self.push_variant(ast_node.iterator)
        ast_node.block.add_variant(ast_node.iterator.name())
        self.visit_children(ast_node)

    def visit_ArrayAccess(self, ast_node):
        # For array accesses, we only want to include the array name, and not
        # the index that is also present in the access node
        self.visit(ast_node.array)

    def visit_Array(self, ast_node):
        self.push_variant(ast_node)

    # TODO: Array should be enough
    def visit_ArrayND(self, ast_node):
        self.push_variant(ast_node)

    def visit_Iter(self, ast_node):
        self.push_variant(ast_node)

    def visit_Property(self, ast_node):
        self.push_variant(ast_node)

    def visit_ContactProperty(self, ast_node):
        self.push_variant(ast_node)

    def visit_FeatureProperty(self, ast_node):
        self.push_variant(ast_node)

    def visit_PropertyAccess(self, ast_node):
        # For property accesses, we only want to include the property name, and not
        # the index that is also present in the access node
        self.visit(ast_node.prop)

    def visit_ContactPropertyAccess(self, ast_node):
        # For property accesses, we only want to include the property name, and not
        # the index that is also present in the access node
        self.visit(ast_node.contact_prop)

    def visit_FeaturePropertyAccess(self, ast_node):
        # For property accesses, we only want to include the property name, and not
        # the index that is also present in the access node
        self.visit(ast_node.feature_prop)

    def visit_Var(self, ast_node):
        self.push_variant(ast_node)


class DetermineParentBlocks(Visitor):
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
            self.clear_visited_nodes()
            self.visit_children(s)

        self.block_stack.pop()

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

    def visit_ScalarOp(self, ast_node):
        self.visit_children(ast_node)
        self.update_ownership(ast_node)

    def visit_Select(self, ast_node):
        self.visit_children(ast_node)
        self.update_ownership(ast_node)

    def visit_VectorOp(self, ast_node):
        self.visit_children(ast_node)
        self.update_ownership(ast_node)
