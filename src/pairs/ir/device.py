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
    def __init__(self, sim, array, ctx, action):
        super().__init__(sim)
        self._array = array
        self._ctx = ctx
        self._action = action
        self.sim.add_statement(self)

    def array(self):
        return self._array

    def context(self):
        return self._ctx

    def action(self):
        return self._action

    def children(self):
        return [self._array]


class CopyProperty(ASTNode):
    def __init__(self, sim, prop, ctx, action):
        super().__init__(sim)
        self._prop = prop
        self._ctx = ctx
        self._action = action
        self.sim.add_statement(self)

    def prop(self):
        return self._prop

    def context(self):
        return self._ctx

    def action(self):
        return self._action

    def children(self):
        return [self._prop]


class CopyContactProperty(ASTNode):
    def __init__(self, sim, prop, ctx, write):
        super().__init__(sim)
        self._contact_prop = prop
        self._ctx = ctx
        self._action = action
        self.sim.add_statement(self)

    def contact_prop(self):
        return self._contact_prop

    def context(self):
        return self._ctx

    def action(self):
        return self._action

    def children(self):
        return [self._prop]


class CopyVar(ASTNode):
    def __init__(self, sim, variable, ctx, action):
        super().__init__(sim)
        self._variable = variable
        self._ctx = ctx
        self._action = action
        self.sim.add_statement(self)

    def variable(self):
        return self._variable

    def context(self):
        return self._ctx

    def action(self):
        return self._action

    def children(self):
        return [self._variable]
