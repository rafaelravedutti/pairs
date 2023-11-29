from functools import reduce
import operator
from pairs.ir.ast_node import ASTNode
from pairs.ir.scalars import ScalarOp
from pairs.ir.sizeof import Sizeof


class HostRef(ASTNode):
    def __init__(self, sim, elem):
        super().__init__(sim)
        self.elem = elem

    def type(self):
        return self.elem.type()

    def children(self):
        return [self.elem]


class DeviceStaticRef(ASTNode):
    def __init__(self, sim, elem):
        super().__init__(sim)
        self.elem = elem

    def type(self):
        return self.elem.type()

    def children(self):
        return [self.elem]


class CopyArray(ASTNode):
    def __init__(self, sim, array, ctx, write):
        super().__init__(sim)
        self.array = array
        self.ctx = ctx
        self.write = write
        self.sim.add_statement(self)

    def context(self):
        return self.ctx

    def children(self):
        return [self.array]


class CopyProperty(ASTNode):
    def __init__(self, sim, prop, ctx, write):
        super().__init__(sim)
        self.prop = prop
        self.ctx = ctx
        self.write = write
        self.sim.add_statement(self)

    def context(self):
        return self.ctx

    def children(self):
        return [self.prop]


class CopyContactProperty(ASTNode):
    def __init__(self, sim, prop, ctx, write):
        super().__init__(sim)
        self.contact_prop = prop
        self.ctx = ctx
        self.write = write
        self.sim.add_statement(self)

    def context(self):
        return self.ctx

    def children(self):
        return [self.prop]


class CopyVar(ASTNode):
    def __init__(self, sim, variable, ctx):
        super().__init__(sim)
        self.variable = variable
        self.ctx = ctx
        self.sim.add_statement(self)

    def context(self):
        return self.ctx

    def children(self):
        return [self.variable]
