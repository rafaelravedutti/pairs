from pairs.ir.arrays import ArrayAccess
from pairs.ir.features import FeaturePropertyAccess
from pairs.ir.loops import For, While
from pairs.ir.math import MathFunction
from pairs.ir.matrices import Matrix, MatrixOp
from pairs.ir.mutator import Mutator
from pairs.ir.properties import PropertyAccess, ContactPropertyAccess
from pairs.ir.quaternions import Quaternion, QuaternionOp
from pairs.ir.scalars import ScalarOp
from pairs.ir.select import Select
from pairs.ir.vectors import Vector, VectorOp


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
        elems_to_check = (
            ArrayAccess,
            ContactPropertyAccess,
            FeaturePropertyAccess,
            MathFunction,
            Matrix,
            MatrixOp,
            PropertyAccess,
            Quaternion,
            QuaternionOp,
            ScalarOp,
            Select,
            Vector,
            VectorOp
        )

        if self.loops and isinstance(ast_node.elem, elems_to_check):
            last_loop = self.loops[-1]
            loop_lifts = self.lifts[id(last_loop)]
            #print(f"id = {ast_node.elem.id()}, variants = {last_loop.block.variants}, terminals = {ast_node.elem.terminals}")
            if not last_loop.block.variants.intersection(ast_node.elem.terminals) and \
               not last_loop.is_kernel_candidate():
                found = False
                for lifted_decls in loop_lifts:
                    if ast_node.elem == lifted_decls.elem:
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
