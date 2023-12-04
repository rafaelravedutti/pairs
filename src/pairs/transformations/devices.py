import math
from pairs.ir.actions import Actions
from pairs.ir.block import Block
from pairs.ir.branches import Branch, Filter
from pairs.ir.cast import Cast
from pairs.ir.contexts import Contexts
from pairs.ir.device import CopyArray, CopyContactProperty, CopyProperty, CopyVar, DeviceStaticRef, HostRef
from pairs.ir.functions import Call_Void
from pairs.ir.kernel import Kernel, KernelLaunch
from pairs.ir.lit import Lit
from pairs.ir.loops import For
from pairs.ir.module import ModuleCall
from pairs.ir.mutator import Mutator
from pairs.ir.scalars import ScalarOp
from pairs.ir.timers import Timers
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
                    new_stmts += [
                        Call_Void(ast_node.sim, "pairs::start_timer", [Timers.DeviceTransfers])
                    ]

                    for array, action in s.module.arrays().items():
                        new_stmts += [CopyArray(s.sim, array, copy_context, action)]

                    for prop, action in s.module.properties().items():
                        new_stmts += [CopyProperty(s.sim, prop, copy_context, action)]

                    for contact_prop, action in s.module.contact_properties().items():
                        new_stmts += [CopyContactProperty(s.sim, contact_prop, copy_context, action)]

                    if self.module_resizes[s.module] and s.module.run_on_device:
                        new_stmts += [CopyArray(s.sim, s.sim.resizes, Contexts.Device, Actions.Ignore)]

                    if s.module.run_on_device:
                        for var, action in s.module.variables().items():
                            if action != Actions.ReadOnly and var.device_flag:
                                new_stmts += [CopyVar(s.sim, var, Contexts.Device, action)]

                    new_stmts += [
                        Call_Void(ast_node.sim, "pairs::stop_timer", [Timers.DeviceTransfers])
                    ]

                new_stmts.append(s)

                if isinstance(s, ModuleCall):
                    if s.module.run_on_device:
                        new_stmts += [
                            Call_Void(ast_node.sim, "pairs::start_timer", [Timers.DeviceTransfers])
                        ]

                        for var, action in s.module.variables().items():
                            if action != Actions.ReadOnly and var.device_flag:
                                new_stmts += [CopyVar(s.sim, var, Contexts.Host, action)]

                        if self.module_resizes[s.module]:
                            new_stmts += [CopyArray(s.sim, s.sim.resizes, Contexts.Host, Actions.Ignore)]
                        new_stmts += [
                            Call_Void(ast_node.sim, "pairs::stop_timer", [Timers.DeviceTransfers])
                        ]

        ast_node.stmts = new_stmts
        return ast_node


class AddDeviceKernels(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)
        self._module_name = None
        self._kernel_id = 0

    def create_kernel(self, sim, iterator, rmax, block):
        kernel_name = f"{self._module_name}_kernel{self._kernel_id}"
        kernel = sim.find_kernel_by_name(kernel_name)

        if kernel is None:
            kernel_body = Filter(sim, ScalarOp.inline(iterator < rmax.copy(True)), block)
            kernel = Kernel(sim, kernel_name, kernel_body, iterator)
            self._kernel_id += 1

        return kernel

    def mutate_Module(self, ast_node):
        if ast_node.run_on_device:
            self._module_name = ast_node.name
            self._kernel_id = 0

            new_stmts = []
            for stmt in ast_node._block.stmts:
                if stmt is not None:
                    if isinstance(stmt, For) and stmt.is_kernel_candidate():
                        kernel = self.create_kernel(ast_node.sim, stmt.iterator, stmt.max, stmt.block)
                        new_stmts.append(
                            KernelLaunch(ast_node.sim, kernel, stmt.iterator, stmt.min, stmt.max))

                    else:
                        if isinstance(stmt, Branch):
                            stmt = self.check_and_mutate_branch(stmt)

                        new_stmts.append(stmt)

            ast_node._block.stmts = new_stmts

        ast_node._block = self.mutate(ast_node._block)
        return ast_node

    def check_and_mutate_branch(self, ast_node):
        new_stmts = []
        for stmt in ast_node.block_if.stmts:
            if stmt is not None:
                if isinstance(stmt, For) and stmt.is_kernel_candidate():
                    kernel = self.create_kernel(ast_node.sim, stmt.iterator, stmt.max, stmt.block)
                    new_stmts.append(
                        KernelLaunch(ast_node.sim, kernel, stmt.iterator, stmt.min, stmt.max))

                else:
                    new_stmts.append(stmt)

        ast_node.block_if.stmts = new_stmts

        if ast_node.block_else is not None:
            new_stmts = []
            for stmt in ast_node.block_else.stmts:
                if stmt is not None:
                    if isinstance(stmt, For) and stmt.is_kernel_candidate():
                        kernel = self.create_kernel(ast_node.sim, stmt.iterator, stmt.max, stmt.block)
                        new_stmts.append(
                            KernelLaunch(ast_node.sim, kernel, stmt.iterator, stmt.min, stmt.max))

                    else:
                        new_stmts.append(stmt)

            ast_node.block_else.stmts = new_stmts

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

    def mutate_FeatureProperty(self, ast_node):
        if self.device_context:
            self.module_stack[-1].add_host_reference(ast_node)
            return HostRef(ast_node.sim, ast_node)

        return ast_node

    def mutate_ContactProperty(self, ast_node):
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
        ast_node._threads_per_block = self.mutate(ast_node._threads_per_block)
        ast_node._nblocks = self.mutate(ast_node._nblocks)
        return ast_node

    def mutate_Property(self, ast_node):
        if self.device_context:
            self.module_stack[-1].add_host_reference(ast_node)
            return HostRef(ast_node.sim, ast_node)

        return ast_node


class AddDeviceReferencesToModules(Mutator):
    def __init__(self, ast=None):
        super().__init__(ast)
        self.kernel_context = False
        self.within_decl = False
        self.add_reference = False
        self.declared_objects = []

    def must_add_reference(self, ast_node):
        return id(ast_node) not in self.declared_objects and self.kernel_context and \
               (ast_node.inlined is True or self.within_decl)

    def mutate_ArrayAccess(self, ast_node):
        if isinstance(ast_node.array, (DeviceStaticRef, HostRef)):
            return ast_node

        _add_reference = self.add_reference
        self.add_reference = ast_node.array.is_static() and self.must_add_reference(ast_node)
        ast_node.array = self.mutate(ast_node.array)
        self.add_reference = _add_reference
        return ast_node

    def mutate_ArrayStatic(self, ast_node):
        if self.add_reference:
            return DeviceStaticRef(ast_node.sim, ast_node)

        return ast_node

    def mutate_FeaturePropertyAccess(self, ast_node):
        _add_reference = self.add_reference
        self.add_reference = self.must_add_reference(ast_node)
        ast_node.feature_prop = self.mutate(ast_node.feature_prop)
        self.add_reference = _add_reference
        return ast_node

    def mutate_FeatureProperty(self, ast_node):
        if self.add_reference:
            return DeviceStaticRef(ast_node.sim, ast_node)

        return ast_node

    def mutate_ContactPropertyAccess(self, ast_node):
        _add_reference = self.add_reference
        self.add_reference = self.must_add_reference(ast_node)
        ast_node.feature_prop = self.mutate(ast_node.contact_prop)
        self.add_reference = _add_reference
        return ast_node

    def mutate_ContactProperty(self, ast_node):
        if self.add_reference:
            return DeviceStaticRef(ast_node.sim, ast_node)

        return ast_node

    def mutate_DeviceStaticRef(self, ast_node):
        return ast_node

    def mutate_Decl(self, ast_node):
        _within_decl = self.within_decl
        self.within_decl = True
        ast_node.elem = self.mutate(ast_node.elem)
        self.declared_objects.append(id(ast_node.elem))
        self.within_decl = _within_decl
        return ast_node

    def mutate_HostRef(self, ast_node):
        return ast_node

    def mutate_Kernel(self, ast_node):
        _kernel_context = self.kernel_context
        self.kernel_context = True
        ast_node._block = self.mutate(ast_node._block)
        self.kernel_context = _kernel_context
        return ast_node
