import math
from pairs.ir.assign import Assign
from pairs.ir.bin_op import BinOp
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.cast import Cast
from pairs.ir.contexts import Contexts
from pairs.ir.device import ClearArrayFlag, ClearPropertyFlag, CopyArray, CopyProperty, SetArrayFlag, SetPropertyFlag, HostRef
from pairs.ir.kernel import Kernel, KernelLaunch
from pairs.ir.lit import Lit
from pairs.ir.loops import For
from pairs.ir.module import ModuleCall
from pairs.ir.mutator import Mutator
from pairs.ir.types import Types


class AddDeviceCopies(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.module_resizes = None

    def set_ast(self, ast):
        super().set_ast(ast)

    def set_data(self, data):
        self.module_resizes = data[0]

    def mutate_Block(self, ast_node):
        new_stmts = []
        stmts = [self.mutate(s) for s in ast_node.stmts]

        for s in stmts:
            if s is not None:
                if isinstance(s, ModuleCall):
                    copy_context = Contexts.Device if s.module.run_on_device else Contexts.Host
                    clear_context = Contexts.Host if s.module.run_on_device else Contexts.Device

                    for a in s.module.arrays_to_synchronize():
                        new_stmts += [CopyArray(s.sim, a, copy_context)]

                    for p in s.module.properties_to_synchronize():
                        new_stmts += [CopyProperty(s.sim, p, copy_context)]

                    for a in s.module.write_arrays():
                        new_stmts += [SetArrayFlag(s.sim, a, copy_context), ClearArrayFlag(s.sim, a, clear_context)]

                    for p in s.module.write_properties():
                        new_stmts += [SetPropertyFlag(s.sim, p, copy_context), ClearPropertyFlag(s.sim, p, clear_context)]

                    if self.module_resizes[s.module] and s.module.run_on_device:
                        new_stmts += [CopyArray(s.sim, s.sim.resizes, Contexts.Device)]

                new_stmts.append(s)

                if isinstance(s, ModuleCall) and self.module_resizes[s.module] and s.module.run_on_device:
                    new_stmts += [CopyArray(s.sim, s.sim.resizes, Contexts.Host)]

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
                        kernel_name = f"{ast_node.name}_kernel{kernel_id}"
                        kernel = ast_node.sim.find_kernel_by_name(kernel_name)
                        if kernel is None:
                            kernel_body = Filter(ast_node.sim, BinOp.inline(s.iterator < s.max), s.block)
                            kernel = Kernel(ast_node.sim, kernel_name, kernel_body, s.iterator)
                            kernel_id += 1

                        new_stmts.append(KernelLaunch(ast_node.sim, kernel, s.iterator, s.min, s.max))

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

    def mutate_Decl(self, ast_node):
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
