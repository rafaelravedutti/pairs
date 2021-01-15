from ast.loops import For, While
from ast.mutator import Mutator
from ast.visitor import Visitor


class SetBlockVariants(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        self.in_assignment = None

    def mutate_Assign(self, ast_node):
        self.in_assignment = ast_node if ast_node.parent_block is not None else None
        for dest in ast_node.destinations():
            self.mutate(dest)
        self.in_assignment = None
        return ast_node

    def mutate_For(self, ast_node):
        ast_node.block.add_variant(id(ast_node.iterator))
        ast_node.iterator = self.mutate(ast_node.iterator)
        ast_node.block = self.mutate(ast_node.block)
        return ast_node

    def mutate_Array(self, ast_node):
        if self.in_assignment is not None:
            self.in_assignment.parent_block.add_variant(id(ast_node))

        return ast_node

    def mutate_Iter(self, ast_node):
        if self.in_assignment is not None:
            self.in_assignment.parent_block.add_variant(id(ast_node))

        return ast_node

    def mutate_Property(self, ast_node):
        if self.in_assignment is not None:
            self.in_assignment.parent_block.add_variant(id(ast_node))

        return ast_node

    def mutate_Variable(self, ast_node):
        if self.in_assignment is not None:
            self.in_assignment.parent_block.add_variant(id(ast_node))

        return ast_node


class SetParentBlock(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.blocks = []

    def current_block(self):
        return self.blocks[-1] if self.blocks else None

    def visit_Assign(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.visit_children(ast_node)

    def visit_Block(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.blocks.append(ast_node)
        self.visit_children(ast_node)
        self.blocks.pop()

    def visit_BinOpDef(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.visit_children(ast_node)

    def visit_Branch(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.visit_children(ast_node)

    def visit_Filter(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.visit_children(ast_node)

    def visit_For(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.visit_children(ast_node)

    def visit_ParticleFor(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.visit_children(ast_node)

    def visit_Malloc(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.visit_children(ast_node)

    def visit_Realloc(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.visit_children(ast_node)

    def visit_While(self, ast_node):
        ast_node.parent_block = self.current_block()
        self.visit_children(ast_node)

    def get_loop_parent_block(self, ast_node):
        assert isinstance(ast_node, (For, While)), "Node must be a loop!"
        loop_id = id(ast_node)
        return self.parents[loop_id] if loop_id in self.parents else None


class SetBinOpTerminals(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.bin_ops = []

    def visit_BinOp(self, ast_node):
        self.bin_ops.append(ast_node)
        self.visit_children(ast_node)
        self.bin_ops.pop()

    def visit_Array(self, ast_node):
        for bin_op in self.bin_ops:
            bin_op.add_terminal(id(ast_node))

    def visit_Iter(self, ast_node):
        for bin_op in self.bin_ops:
            bin_op.add_terminal(id(ast_node))

    def visit_Property(self, ast_node):
        for bin_op in self.bin_ops:
            bin_op.add_terminal(id(ast_node))

    def visit_Variable(self, ast_node):
        for bin_op in self.bin_ops:
            bin_op.add_terminal(id(ast_node))


class LICM(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        self.lifts = {}
        self.loops = []

    def mutate_For(self, ast_node):
        self.lifts[id(ast_node)] = []
        self.loops.append(ast_node)
        ast_node.iterator = self.mutate(ast_node.iterator)
        ast_node.block = self.mutate(ast_node.block)
        self.loops.pop()
        return ast_node

    def mutate_BinOpDef(self, ast_node):
        if self.loops:
            last_loop = self.loops[-1]
            print(f"Checking lifting for {ast_node.id()}")
            if not last_loop.block.variants.intersect(ast_node.bin_op.terminals):
                self.lifts[id(last_loop)].append(ast_node)
                print(f"Lifting {ast_node.id()}")
                return None

        return ast_node

    def mutate_Block(self, ast_node):
        new_stmts = []
        stmts = self.mutate(ast_node.stmts)

        for s in stmts:
            if s is not None:
                s_id = id(s)
                if isinstance(s, (For, While)) and s_id in self.lifts:
                    new_stmts = new_stmts + self.lifts[s_id]

                new_stmts.append(s)

        ast_node.stmts = new_stmts
        return ast_node


def move_loop_invariant_code(ast):
    set_parent_block = SetParentBlock(ast)
    set_parent_block.visit()
    set_block_variants = SetBlockVariants(ast)
    set_block_variants.mutate()
    set_bin_op_terminals = SetBinOpTerminals(ast)
    set_bin_op_terminals.visit()
    licm = LICM(ast)
    licm.mutate()
