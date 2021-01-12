from ast.arrays import ArrayAccess
from ast.assign import Assign
from ast.bin_op import BinOp, BinOpDef
from ast.block import Block
from ast.branches import Branch
from ast.cast import Cast
from ast.loops import For, While
from ast.math import Sqrt
from ast.memory import Malloc, Realloc
from ast.select import Select
from sim.timestep import Timestep


class Mutator:
    def __init__(self, ast, max_depth=0):
        self.ast = ast
        self.max_depth = 0

    def mutate(self, ast_node=None):
        if ast_node is None:
            ast_node = self.ast

        if isinstance(ast_node, ArrayAccess):
            return self.mutate_ArrayAccess(ast_node)

        elif isinstance(ast_node, Assign):
            return self.mutate_Assign(ast_node)

        elif isinstance(ast_node, BinOp):
            return self.mutate_BinOp(ast_node)

        elif isinstance(ast_node, BinOpDef):
            return self.mutate_BinOpDef(ast_node)

        elif isinstance(ast_node, Block):
            return self.mutate_Block(ast_node)

        elif isinstance(ast_node, Branch):
            return self.mutate_Branch(ast_node)

        elif isinstance(ast_node, Cast):
            return self.mutate_Cast(ast_node)

        elif isinstance(ast_node, For):
            return self.mutate_For(ast_node)

        elif isinstance(ast_node, Malloc):
            return self.mutate_Malloc(ast_node)

        elif isinstance(ast_node, Realloc):
            return self.mutate_Realloc(ast_node)

        elif isinstance(ast_node, Select):
            return self.mutate_Select(ast_node)

        elif isinstance(ast_node, Sqrt):
            return self.mutate_Sqrt(ast_node)

        elif isinstance(ast_node, Timestep):
            return self.mutate_Timestep(ast_node)

        elif isinstance(ast_node, While):
            return self.mutate_While(ast_node)

        return ast_node

    def mutate_ArrayAccess(self, ast_node):
        ast_node.array = self.mutate(ast_node.array)
        ast_node.indexes = [self.mutate(i) for i in ast_node.indexes]

        if ast_node.index is not None:
            ast_node.index = self.mutate(ast_node.index)

        return ast_node 

    def mutate_Assign(self, ast_node):
        ast_node.assignments = [
            (self.mutate(ast_node.assignments[i][0]), self.mutate(ast_node.assignments[i][1]))
            for i in range(0, len(ast_node.assignments))
        ]
        return ast_node

    def mutate_BinOp(self, ast_node):
        ast_node.lhs = self.mutate(ast_node.lhs)
        ast_node.rhs = self.mutate(ast_node.rhs)
        ast_node.bin_op_vector_index_mapping = {i: self.mutate(e) for i, e in ast_node.bin_op_vector_index_mapping.items()}
        return ast_node

    def mutate_BinOpDef(self, ast_node):
        ast_node.bin_op = self.mutate(ast_node.bin_op)
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

    def mutate_For(self, ast_node):
        ast_node.iterator = self.mutate(ast_node.iterator)
        ast_node.block = self.mutate(ast_node.block)
        return ast_node

    def mutate_Malloc(self, ast_node):
        ast_node.array = self.mutate(ast_node.array)
        ast_node.size = self.mutate(ast_node.size)
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

    def mutate_While(self, ast_node):
        ast_node.cond = self.mutate(ast_node.cond)
        ast_node.block = self.mutate(ast_node.block)
        return ast_node
