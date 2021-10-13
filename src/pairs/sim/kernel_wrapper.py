from pairs.ir.block import Block, KernelBlock


class KernelWrapper():
    def __init__(self, sim):
        self.sim = sim
        self.kernels = Block(sim, [])

    def add_kernel_block(self, block):
        self.kernels = Block.merge_blocks(self.kernels, KernelBlock(self.sim, block))

    def lower(self):
        return self.kernels
