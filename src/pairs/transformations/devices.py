import math
from pairs.ir.assign import Assign
from pairs.ir.bin_op import BinOp
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.cast import Cast
from pairs.ir.device import ClearArrayDeviceFlag, ClearArrayHostFlag, ClearPropertyDeviceFlag, ClearPropertyHostFlag
from pairs.ir.device import CopyArrayToDevice, CopyArrayToHost, CopyPropertyToDevice, CopyPropertyToHost, HostRef
from pairs.ir.kernel import Kernel, KernelLaunch
from pairs.ir.lit import Lit
from pairs.ir.loops import For
from pairs.ir.module import ModuleCall
from pairs.ir.mutator import Mutator
from pairs.ir.types import Types


class AddDeviceCopies(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)

    def set_ast(self, ast):
        super().set_ast(ast)

    def mutate_Block(self, ast_node):
        new_stmts = []
        stmts = [self.mutate(s) for s in ast_node.stmts]

        for s in stmts:
            if s is not None:
                if isinstance(s, ModuleCall):
                    for a in s.module.arrays_to_synchronize():
                        if s.module.run_on_device:
                            new_stmts += [CopyArrayToDevice(s.sim, a)]
                        else:
                            new_stmts += [CopyArrayToHost(s.sim, a)]

                    for p in s.module.properties_to_synchronize():
                        if s.module.run_on_device:
                            new_stmts += [CopyPropertyToDevice(s.sim, p)]
                        else:
                            new_stmts += [CopyPropertyToHost(s.sim, p)]

                    for a in s.module.write_arrays():
                        if s.module.run_on_device:
                            new_stmts += [ClearArrayHostFlag(s.sim, a)]
                        else:
                            new_stmts += [ClearArrayDeviceFlag(s.sim, a)]

                    for p in s.module.write_properties():
                        if s.module.run_on_device:
                            new_stmts += [ClearPropertyHostFlag(s.sim, p)]
                        else:
                            new_stmts += [ClearPropertyDeviceFlag(s.sim, p)]

                new_stmts.append(s)

        ast_node.stmts = new_stmts
        return ast_node


class AddDeviceKernels(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)

    def mutate_Module(self, ast_node):
        ast_node._block = self.mutate(ast_node._block)

        if ast_node.run_on_device:
            new_stmts = []
            kernel_id = 0
            for s in ast_node._block.stmts:
                if s is not None:
                    if isinstance(s, For) and (not isinstance(s.min, Lit) or not isinstance(s.max, Lit)):
                        kernel_block = Filter(ast_node.sim, BinOp.inline(s.iterator < s.max), s.block)
                        kernel = Kernel(ast_node.sim, f"{ast_node.name}_kernel{kernel_id}", kernel_block, s.iterator)
                        new_stmts.append(KernelLaunch(ast_node.sim, kernel, s.iterator, s.min, s.max))
                        kernel_id += 1
                    else:
                        new_stmts.append(s)

            ast_node._block.stmts = new_stmts

        return ast_node


class AddHostReferencesToModules(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.module_stack = []
        self.device_context = False

    def mutate_Array(self, ast_node):
        if self.device_context:
            self.module_stack[-1].add_host_reference(ast_node)
            return HostRef(ast_node.sim, ast_node)

        return ast_node

    def mutate_ArrayND(self, ast_node):
        if self.device_context:
            self.module_stack[-1].add_host_reference(ast_node)
            return HostRef(ast_node.sim, ast_node)

        return ast_node

    def mutate_ArrayStatic(self, ast_node):
        if self.device_context:
            self.module_stack[-1].add_host_reference(ast_node)
            return HostRef(ast_node.sim, ast_node)

        return ast_node

    def mutate_HostRef(self, ast_node):
        return ast_node

    def mutate_Module(self, ast_node):
        _device_context = self.device_context
        self.device_context = self.device_context or ast_node.run_on_device
        self.module_stack.append(ast_node)
        ast_node._block = self.mutate(ast_node._block)
        self.module_stack.pop()
        self.device_context = _device_context
        return ast_node

    def mutate_Kernel(self, ast_node):
        return ast_node

    def mutate_KernelLaunch(self, ast_node):
        return ast_node

    def mutate_Property(self, ast_node):
        if self.device_context:
            self.module_stack[-1].add_host_reference(ast_node)
            return HostRef(ast_node.sim, ast_node)

        return ast_node
