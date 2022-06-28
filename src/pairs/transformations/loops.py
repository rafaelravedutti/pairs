from pairs.ir.arrays import ArrayAccess
from pairs.ir.bin_op import BinOp
from pairs.ir.loops import For, While
from pairs.ir.mutator import Mutator
from pairs.ir.properties import PropertyAccess


class LICM(Mutator):
    def __init__(self, ast=None):
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

    def mutate_Decl(self, ast_node):
        if self.loops and isinstance(ast_node.elem, (BinOp, ArrayAccess, PropertyAccess)):
            last_loop = self.loops[-1]
            loop_lifts = self.lifts[id(last_loop)]
            #print(f"variants = {last_loop.block.variants}, terminals = {ast_node.elem.terminals}")
            if not last_loop.block.variants.intersection(ast_node.elem.terminals):
                found = False
                for d in loop_lifts:
                    if ast_node.elem == d.elem:
                        found = True

                if not found:
                    #print(f'lifting {ast_node.elem.id()}')
                    loop_lifts.append(ast_node)

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
