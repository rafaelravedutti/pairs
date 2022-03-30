import math
from pairs.ir.assign import Assign
from pairs.ir.bin_op import BinOp
from pairs.ir.block import Block
from pairs.ir.branches import Filter
from pairs.ir.device import CopyToDevice, CopyToHost
from pairs.ir.kernel import Kernel, KernelLaunch
from pairs.ir.lit import Lit
from pairs.ir.loops import For
from pairs.ir.module import ModuleCall
from pairs.ir.mutator import Mutator
from pairs.ir.types import Types


class AddDeviceCopies(Mutator):
    def __init__(self, ast):
        super().__init__(ast)
        nprops = len(ast.sim.properties.all())
        self.nflags = math.ceil(nprops / 64.0)
        self.prop_hflags = ast.sim.add_static_array('prop_hflags', self.nflags, Types.UInt64, init_value=0xffffffffffffffff)
        self.prop_dflags = ast.sim.add_static_array('prop_dflags', self.nflags, Types.UInt64, init_value=0)

    def mutate_Block(self, ast_node):
        new_stmts = []
        stmts = [self.mutate(s) for s in ast_node.stmts]

        for s in stmts:
            if s is not None:
                if isinstance(s, ModuleCall):
                    sync_flags = [0] * self.nflags
                    dirty_flags = [0] * self.nflags

                    for p in s.module.properties_to_synchronize():
                        flag_index = p.id() // 64
                        bit = p.id() % 64

                        if s.module.run_on_device:
                            new_stmts += [
                                Filter(s.sim,
                                    BinOp.cmp(self.prop_dflags[flag_index] & (1 << bit), 0),
                                    Block(s.sim, CopyToDevice(s.sim, p)))]
                        else:
                            new_stmts += [
                                Filter(s.sim,
                                    BinOp.cmp(self.prop_hflags[flag_index] & (1 << bit), 0),
                                    Block(s.sim, CopyToHost(s.sim, p)))]

                        sync_flags[flag_index] |= (1 << bit)

                    for p in s.module.write_properties():
                        flag_index = p.id() // 64
                        bit = p.id() % 64
                        dirty_flags[flag_index] |= (1 << bit)

                    if s.module.run_on_device:
                        new_stmts += \
                            [Assign(s.sim, self.prop_dflags[i], self.prop_dflags[i] | sync_flags[i]) for i in range(self.nflags)] + \
                            [Assign(s.sim, self.prop_hflags[i], self.prop_hflags[i] & ~dirty_flags[i]) for i in range(self.nflags)]
                    else:
                        new_stmts += \
                            [Assign(s.sim, self.prop_hflags[i], self.prop_hflags[i] | sync_flags[i]) for i in range(self.nflags)] + \
                            [Assign(s.sim, self.prop_dflags[i], self.prop_dflags[i] & ~dirty_flags[i]) for i in range(self.nflags)]

                new_stmts.append(s)

        ast_node.stmts = new_stmts
        return ast_node


class AddDeviceKernels(Mutator):
    def __init__(self, ast):
        super().__init__(ast)

    def mutate_Module(self, ast_node):
        ast_node._block = self.mutate(ast_node._block)

        if ast_node.run_on_device:
            new_stmts = []
            kernel_id = 0
            for s in ast_node._block.stmts:
                if s is not None:
                    if isinstance(s, For) and (not isinstance(s.min, Lit) or not isinstance(s.max, Lit)):
                        kernel = Kernel(ast_node.sim, f"{ast_node.name}_kernel{kernel_id}", s.block)
                        new_stmts.append(KernelLaunch(ast_node.sim, kernel, s.iterator, s.min, s.max))
                        kernel_id += 1
                    else:
                        new_stmts.append(s)

            ast_node._block_stmts = new_stmts

        return ast_node
