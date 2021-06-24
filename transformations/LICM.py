from ir.bin_op import BinOp
from ir.loops import For, While
from ir.mutator import Mutator
from ir.visitor import Visitor


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

        # For property accesses, we only want to include the property name, and not
        # the index that is also present in the expression
        if not ast_node.is_property_access():
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

    def visit_BinOpDef(self, ast_node):
        self.set_parent_block(ast_node)

    def visit_Branch(self, ast_node):
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


class SetBinOpTerminals(Visitor):
    def __init__(self, ast):
        super().__init__(ast)
        self.bin_ops = []

    def push_terminal(self, ast_node):
        for bin_op in self.bin_ops:
            bin_op.add_terminal(ast_node.name())

    def visit_BinOp(self, ast_node):
        self.bin_ops.append(ast_node)
        self.visit_children(ast_node)
        self.bin_ops.pop()

    def visit_Array(self, ast_node):
        self.push_terminal(ast_node)

    # TODO: Array should be enough
    def visit_ArrayND(self, ast_node):
        self.push_terminal(ast_node)

    def visit_Iter(self, ast_node):
        self.push_terminal(ast_node)

    def visit_Property(self, ast_node):
        self.push_terminal(ast_node)

    def visit_Var(self, ast_node):
        self.push_terminal(ast_node)


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

    def mutate_While(self, ast_node):
        self.lifts[id(ast_node)] = []
        self.loops.append(ast_node)
        ast_node.cond = self.mutate(ast_node.cond)
        ast_node.block = self.mutate(ast_node.block)
        self.loops.pop()
        return ast_node

    def mutate_BinOpDef(self, ast_node):
        if self.loops and isinstance(ast_node.bin_op, BinOp):
            last_loop = self.loops[-1]
            #print(f"variants = {last_loop.block.variants}, terminals = {ast_node.bin_op.terminals}")
            if not last_loop.block.variants.intersection(ast_node.bin_op.terminals):
                #print(f'lifting {ast_node.bin_op.id()}')
                self.lifts[id(last_loop)].append(ast_node)
                return None

        return ast_node

    def mutate_Block(self, ast_node):
        new_stmts = []
        stmts = [self.mutate(s) for s in ast_node.stmts]

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
