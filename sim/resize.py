from ast.block import BlockAST
from ast.branches import BranchAST
from ast.data_types import Type_Int
from ast.loops import WhileAST

class Resize:
    def __init__(self, sim, capacity_var, arrays, body, grow_fn=None):
        self.sim = sim
        self.capacity_var = capacity_var
        self.arrays = [arrays] if not isinstance(arrays, list) else arrays
        self.resize_var = self.sim.add_var('resize', Type_Int)
        self.body = body
        self.grow_fn = grow_fn if grow_fn is not None else (lambda x: x * 2)

    def block(self):
        return BlockAST(self.sim, [
            self.resize_var.set(1),
            WhileAST(self.sim, self.resize_var > 0, BlockAST(self.sim, [self.resize_var.set(0)] + self.body + [
                self.capacity_var.set(self.grow_fn(self.resize_var)),
                BranchAST.if_stmt(self.sim, self.resize_var > 0, BlockAST(self.sim, [a.realloc() for a in self.arrays]))
            ])),
        ])

    def check(self, size, body):
        return BranchAST.if_else_stmt(self.sim, size > self.capacity_var, self.resize_var.set(size), body)
