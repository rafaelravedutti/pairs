from functools import reduce
import operator
from pairs.ir.ast_node import ASTNode
from pairs.ir.bin_op import BinOp
from pairs.ir.sizeof import Sizeof


class HostRef(ASTNode):
    def __init__(self, sim, elem):
        super().__init__(sim)
        self.elem = elem

    def type(self):
        return self.elem.type()

    def children(self):
        return [self.elem]


class CopyArrayToDevice(ASTNode):
    def __init__(self, sim, array):
        super().__init__(sim)
        self.array = array

    def children(self):
        return [self.array]


class CopyArrayToHost(ASTNode):
    def __init__(self, sim, array):
        super().__init__(sim)
        self.array = array

    def children(self):
        return [self.array]

class CopyPropertyToDevice(ASTNode):
    def __init__(self, sim, prop):
        super().__init__(sim)
        self.prop = prop

    def children(self):
        return [self.prop]


class CopyPropertyToHost(ASTNode):
    def __init__(self, sim, prop):
        super().__init__(sim)
        self.prop = prop

    def children(self):
        return [self.prop]


class ClearArrayDeviceFlag(ASTNode):
    def __init__(self, sim, array):
        super().__init__(sim)
        self.array = array

    def children(self):
        return [self.array]


class ClearArrayHostFlag(ASTNode):
    def __init__(self, sim, array):
        super().__init__(sim)
        self.array = array

    def children(self):
        return [self.array]


class ClearPropertyDeviceFlag(ASTNode):
    def __init__(self, sim, prop):
        super().__init__(sim)
        self.prop = prop

    def children(self):
        return [self.prop]


class ClearPropertyHostFlag(ASTNode):
    def __init__(self, sim, prop):
        super().__init__(sim)
        self.prop = prop

    def children(self):
        return [self.prop]

