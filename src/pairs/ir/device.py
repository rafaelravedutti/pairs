from functools import reduce
import operator
from pairs.ir.ast_node import ASTNode
from pairs.ir.bin_op import BinOp
from pairs.ir.sizeof import Sizeof


class HostRef(ASTNode):
    def __init__(self, sim, elem):
        super().__init__(sim)
        self.elem = elem
        self.sim.add_statement(self)

    def type(self):
        return self.elem.type()

    def children(self):
        return [self.elem]


class CopyArray(ASTNode):
    def __init__(self, sim, array, ctx):
        super().__init__(sim)
        self.array = array
        self.ctx = ctx
        self.sim.add_statement(self)

    def context(self):
        return self.ctx

    def children(self):
        return [self.array]


class CopyProperty(ASTNode):
    def __init__(self, sim, prop, ctx):
        super().__init__(sim)
        self.prop = prop
        self.ctx = ctx
        self.sim.add_statement(self)

    def context(self):
        return self.ctx

    def children(self):
        return [self.prop]


class ClearArrayFlag(ASTNode):
    def __init__(self, sim, array, ctx):
        super().__init__(sim)
        self.array = array
        self.ctx = ctx
        self.sim.add_statement(self)

    def context(self):
        return self.ctx

    def children(self):
        return [self.array]


class ClearPropertyFlag(ASTNode):
    def __init__(self, sim, prop, ctx):
        super().__init__(sim)
        self.prop = prop
        self.ctx = ctx
        self.sim.add_statement(self)

    def context(self):
        return self.ctx

    def children(self):
        return [self.prop]


class SetArrayFlag(ASTNode):
    def __init__(self, sim, array, ctx):
        super().__init__(sim)
        self.array = array
        self.ctx = ctx
        self.sim.add_statement(self)

    def context(self):
        return self.ctx

    def children(self):
        return [self.array]


class SetPropertyFlag(ASTNode):
    def __init__(self, sim, prop, ctx):
        super().__init__(sim)
        self.prop = prop
        self.ctx = ctx
        self.sim.add_statement(self)

    def context(self):
        return self.ctx

    def children(self):
        return [self.prop]
